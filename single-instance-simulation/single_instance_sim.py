#!/usr/bin/env python3
"""
Single-instance LLM serving discrete-event simulator.
IMPORTANT: synthetic analytical cost model, NOT hardware measurements.
Target scenario: 2x RTX 3090, TP=2, Qwen3-32B (user-provided target).
Replace SyntheticCostModel with a P99 profiling-table predictor for real calibration.
"""
from __future__ import annotations
import argparse, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED=20261111
POLICIES=["fixed256","fixed1024","gpu_threshold","deadline_adaptive","priority_deadline"]
WORKLOADS={
    "low": dict(rate=0.35, prompt_med=512, prompt_sigma=0.8, out_med=64, out_sigma=0.6, burst=False),
    "medium": dict(rate=1.5, prompt_med=2048, prompt_sigma=1.0, out_med=128, out_sigma=0.8, burst=False),
    "high": dict(rate=4.0, prompt_med=8192, prompt_sigma=1.25, out_med=256, out_sigma=1.0, burst=False),
    "overload": dict(rate=8.0, prompt_med=16384, prompt_sigma=1.5, out_med=512, out_sigma=1.2, burst=True),
}
class Req:
    def __init__(self,i,a,p,o):
        self.id=i; self.arr=a; self.prompt=int(p); self.output=int(o)
        self.prefill=0; self.started=None; self.first=None; self.done=None
        self.generated=0; self.itls=[]; self.last_token=None
        self.priority=0.0

class SyntheticCostModel:
    """Illustrative TP=2 cost model; units ms. Deliberately stress-heavy at long contexts."""
    def predict(self,n_active,chunk,cached_context,mode="mixed"):
        dec = (0.0 if n_active==0 else 8.0 + 0.65*n_active + 0.00012*n_active*cached_context)
        pre = (0.0 if chunk<=0 else 3.0 + 0.018*chunk + 0.000002*chunk*cached_context)
        contention = 0.002*chunk*n_active + (0.8 if n_active and chunk else 0.0)
        return max(0.25, dec+pre+contention)

def make_trace(cfg, horizon, seed):
    rng=np.random.default_rng(seed)
    arr=[]; t=0.0
    while t<horizon:
        if cfg["burst"]:
            hot=(int(t//8)%2==0)
            rate=cfg["rate"]*(3.0 if hot else 0.18)
        else: rate=cfg["rate"]
        t += rng.exponential(1.0/max(rate,1e-9))
        if t>=horizon: break
        p=int(np.clip(rng.lognormal(np.log(cfg["prompt_med"]),cfg["prompt_sigma"]),128,100000))
        o=int(np.clip(rng.lognormal(np.log(cfg["out_med"]),cfg["out_sigma"]),16,4096))
        arr.append(Req(len(arr),t,p,o))
    return arr

def percentile(x,q):
    return float(np.percentile(x,q)) if len(x) else float("nan")

def simulate(workload,policy,seed,horizon=45.0):
    cfg=WORKLOADS[workload]; requests=make_trace(cfg,horizon,seed)
    if not requests: return {}
    model=SyntheticCostModel(); now=0.0; idx=0; waiting=[]; active=[]
    kv_peak=0.0; gpu_samples=[]; chunks=[]; iter_times=[]
    ttft_slo={"low":1500,"medium":2500,"high":6000,"overload":10000}[workload]
    tpot_slo={"low":80,"medium":100,"high":150,"overload":220}[workload]
    max_iter=300000
    for it in range(max_iter):
        while idx<len(requests) and requests[idx].arr<=now:
            r=requests[idx]; r.started=now; waiting.append(r); idx+=1
        if not waiting and not active:
            if idx>=len(requests): break
            now=max(now,requests[idx].arr); continue
        if waiting:
            if policy=="priority_deadline":
                waiting.sort(key=lambda r: (-(0.55*(now-r.arr)/max(ttft_slo/1000,0.1)
                    +0.30*(r.prompt/10000)+0.15*(r.output/512)),r.arr))
            else: waiting.sort(key=lambda r:r.arr)
        decode_batch=list(active)
        pre=None; chunk=0
        if waiting:
            pre=waiting[0]
            rem=pre.prompt-pre.prefill
            if policy=="fixed256": chunk=min(256,rem)
            elif policy=="fixed1024": chunk=min(1024,rem)
            elif policy=="gpu_threshold":
                util=(len(active)*0.10 + len(waiting)*0.025)
                chunk=min(1024 if util<0.55 else 256,rem)
            elif policy=="deadline_adaptive":
                if active:
                    slack=min((r.last_token+tpot_slo/1000.0)-now for r in active)
                    lo,hi=0,min(4096,rem)
                    ctx=max((r.prompt for r in active),default=0)
                    while lo<hi:
                        mid=(lo+hi+1)//2
                        if model.predict(len(active),mid,ctx)*1.25 <= max(0,slack*1000): lo=mid
                        else: hi=mid-1
                    chunk=lo
                else: chunk=min(4096,rem)
            elif policy=="priority_deadline": chunk=min(512,rem)
        ctx=max((r.prompt for r in active),default=(pre.prefill if pre else 0))
        dt=model.predict(len(active),chunk,ctx)
        if chunk==0 and not active and waiting:
            chunk=min(128,waiting[0].prompt-waiting[0].prefill); dt=model.predict(0,chunk,0)
        if pre and chunk>0:
            pre.prefill += chunk; chunks.append(chunk)
            if pre.prefill>=pre.prompt:
                waiting.pop(0); pre.first=now+dt/1000.0; pre.last_token=now+dt/1000.0
                pre.generated=1; active.append(pre)
        finished=[]
        for r in decode_batch:
            if r.generated>1: r.itls.append(dt)
            if r.generated>=r.output:
                r.done=now+dt/1000.0; finished.append(r)
            else:
                r.generated+=1; r.last_token=now+dt/1000.0
        for r in finished: active.remove(r)
        now += dt/1000.0
        iter_times.append(dt)
        gpu_samples.append(min(0.99,0.12+0.075*len(active)+0.00012*chunk))
        kv_peak=max(kv_peak,min(0.99,0.08+0.018*len(active)+sum(x.prefill for x in active)/max(1,100000*max(1,len(active)))))
        if idx>=len(requests) and not waiting and not active: break
    completed=[r for r in requests if r.done is not None and r.first is not None]
    ttft=np.array([(r.first-r.arr)*1000 for r in completed])
    tpot=np.array([np.mean(r.itls) if r.itls else 0 for r in completed])
    lat=np.array([(r.done-r.arr)*1000 for r in completed])
    good=sum((r.first-r.arr)*1000<=ttft_slo and (np.mean(r.itls) if r.itls else 0)<=tpot_slo for r in completed)
    duration=max(now,1e-9)
    return dict(workload=workload,policy=policy,seed=seed,offered_rps=len(requests)/horizon,
        arrivals=len(requests),completed=len(completed),completion_ratio=len(completed)/len(requests),
        p50_ttft_ms=percentile(ttft,50),p95_ttft_ms=percentile(ttft,95),p99_ttft_ms=percentile(ttft,99),
        p50_tpot_ms=percentile(tpot,50),p95_tpot_ms=percentile(tpot,95),p99_tpot_ms=percentile(tpot,99),
        request_throughput_rps=len(completed)/duration,goodput_rps=good/duration,
        goodput_attainment=good/max(1,len(completed)),mean_latency_ms=float(np.mean(lat)) if len(lat) else np.nan,
        mean_chunk_tokens=float(np.mean(chunks)) if chunks else 0,chunk_count=len(chunks),
        avg_iteration_ms=float(np.mean(iter_times)) if iter_times else np.nan,
        p99_iteration_ms=percentile(iter_times,99),gpu_util_proxy=float(np.mean(gpu_samples)) if gpu_samples else 0,
        kv_peak_proxy=kv_peak,deadline_miss_rate=float(np.mean(tpot>tpot_slo)) if len(tpot) else np.nan)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizon",type=float,default=45)
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--out",default="results")
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    rows=[]
    for wi,w in enumerate(WORKLOADS):
        for pi,p in enumerate(POLICIES):
            for s in range(args.seeds):
                rows.append(simulate(w,p,SEED+wi*1000+pi*100+s,args.horizon))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(args.out,"runs.csv"),index=False)
    agg=df.groupby(["workload","policy"]).mean(numeric_only=True).reset_index()
    agg.to_csv(os.path.join(args.out,"summary.csv"),index=False)
    metrics=[
        ("p99_ttft_ms","P99 TTFT (ms)"),
        ("p99_tpot_ms","P99 TPOT/TBT (ms)"),
        ("request_throughput_rps","Completed request throughput (req/s)"),
        ("goodput_rps","Goodput (req/s)"),
        ("gpu_util_proxy","GPU utilization proxy"),
        ("mean_chunk_tokens","Mean selected prefill chunk (tokens)"),
    ]
    for col,title in metrics:
        fig,ax=plt.subplots(figsize=(10,5.6))
        for p in POLICIES:
            sub=agg[agg.policy==p].set_index("workload").reindex(WORKLOADS.keys())
            ax.plot(list(WORKLOADS.keys()),sub[col],marker="o",linewidth=2,label=p)
        ax.set_title(title+" — synthetic 2×3090 TP=2 stress simulation")
        ax.set_xlabel("Load regime"); ax.set_ylabel(title); ax.grid(True,alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out,f"curve_{col}.png"),dpi=180); plt.close(fig)
    model=SyntheticCostModel(); rows=[]
    for n in [0,8,16,32,64,96,128]:
        for c in [64,128,256,512,1024,2048,4096]:
            for ctx in [512,4096,16384,65536,100000]:
                rows.append(dict(active_decode=n,chunk=c,cached_context=ctx,iteration_ms=model.predict(n,c,ctx)))
    pd.DataFrame(rows).to_csv(os.path.join(args.out,"synthetic_cost_surface.csv"),index=False)
    fig,ax=plt.subplots(figsize=(10,5.6))
    cost=pd.DataFrame(rows)
    for n in [0,16,32,64,128]:
        q=cost[(cost.active_decode==n)&(cost.cached_context==16384)]
        ax.plot(q.chunk,q.iteration_ms,marker="o",label=f"active decode={n}")
    ax.set_xscale("log",base=2); ax.set_xlabel("Prefill chunk (tokens)"); ax.set_ylabel("Predicted iteration time (ms)")
    ax.set_title("Synthetic runtime cost curve (context=16K)"); ax.grid(True,alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out,"curve_runtime_cost.png"),dpi=180); plt.close(fig)
    print(agg[["workload","policy","p99_ttft_ms","p99_tpot_ms","request_throughput_rps","goodput_rps"]].to_string(index=False))
if __name__=="__main__": main()
