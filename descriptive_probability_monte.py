import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats, optimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability Distribution Learning App", page_icon="📊", layout="wide")
PDFS=["Normal","Truncated normal","Beta","Lognormal","Exponential","Triangular","Uniform"]

def fit_all(x, chosen):
    mn,mx=float(x.min()),float(x.max()); sp=mx-mn; out=[]
    for name in chosen:
        try:
            if name=="Normal": d=stats.norm; p=d.fit(x); k=2
            elif name=="Truncated normal":
                d=stats.truncnorm; lo=max(0,mn); hi=mx
                def fun(z):
                    mu,ls=z; sd=np.exp(ls)
                    return -np.sum(d.logpdf(x,(lo-mu)/sd,(hi-mu)/sd,loc=mu,scale=sd))
                mu,ls=optimize.minimize(fun,[x.mean(),np.log(max(x.std(ddof=1),1e-6))],method="Nelder-Mead").x
                sd=np.exp(ls); p=((lo-mu)/sd,(hi-mu)/sd,mu,sd); k=2
            elif name=="Beta":
                if sp<=0: continue
                d=stats.beta; lo=0 if mn>=0 else mn-.05*sp; hi=mx+max(.05*sp,1e-6)
                p=d.fit(x,floc=lo,fscale=hi-lo); k=2
            elif name=="Lognormal":
                if np.any(x<=0): continue
                d=stats.lognorm; p=d.fit(x,floc=0); k=2
            elif name=="Exponential": d=stats.expon; p=d.fit(x,floc=max(0,mn)); k=1
            elif name=="Triangular": d=stats.triang; p=d.fit(x); k=3
            else: d=stats.uniform; p=(mn,sp); k=2
            ks,pv=stats.kstest(x,d.cdf,args=p); ll=np.sum(d.logpdf(x,*p))
            if np.isfinite(ll): out.append(dict(name=name,dist=d,params=p,ks=ks,pv=pv,aic=2*k-2*ll,bic=k*math.log(len(x))-2*ll))
        except Exception:
            pass
    return out

def simulate(fit,x,n):
    lo=max(0,float(x.min())); hi=float(x.max()); rng=np.random.default_rng(557); parts=[]; got=0; attempts=0
    while got<n and attempts<200:
        z=np.asarray(fit["dist"].rvs(*fit["params"],size=max(5000,n-got),random_state=rng))
        z=z[np.isfinite(z)&(z>=lo)&(z<=hi)]; parts.append(z); got+=len(z); attempts+=1
    if got<n: raise RuntimeError("Could not generate enough accepted draws for the selected distribution.")
    return np.concatenate(parts)[:n]

def fig_show(fig):
    fig.tight_layout(); st.pyplot(fig, clear_figure=True)

st.title("Probability Distribution Learning App")
st.caption("Upload numerical CSV data, compare theoretical distributions, and run Monte Carlo simulation.")

with st.sidebar:
    st.header("1. Load data")
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    st.header("2. Analysis settings")
    chosen=st.multiselect("PDFs to fit",PDFS,default=PDFS)
    bins=st.number_input("Histogram bins",2,100,10,1)
    nsim=st.number_input("Monte Carlo accepted draws",1000,500000,100000,1000)
    hist_color=st.color_picker("Histogram color","#5B7BEA")
    line_color=st.color_picker("Selected fit color","#C00070")

if uploaded is None:
    st.info("Upload a CSV file from the sidebar. The file must contain at least one numeric column and at least five valid observations.")
    st.stop()
try: df=pd.read_csv(uploaded)
except Exception as e:
    st.error(f"CSV error: {e}"); st.stop()
numcols=df.select_dtypes(include=np.number).columns.tolist()
if not numcols:
    st.error("No numeric columns were found in the uploaded CSV."); st.stop()
col=st.sidebar.selectbox("Numeric column",numcols)
x=pd.to_numeric(df[col],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
if len(x)<5:
    st.error("At least five valid numeric values are required."); st.stop()
if not chosen:
    st.error("Select at least one PDF to fit."); st.stop()
with st.spinner("Fitting selected distributions..."):
    fits=fit_all(x,chosen)
if not fits:
    st.error("No selected distribution could be fitted to this column."); st.stop()
best=min(fits,key=lambda z:z["ks"])["name"]
names=[z["name"] for z in fits]
selected=st.sidebar.selectbox("PDF used for display and Monte Carlo",names,index=names.index(best))
fit=next(z for z in fits if z["name"]==selected)
st.success(f"Column: {col} | n={len(x):,} | selected={selected} | KS={fit['ks']:.4f}")

views=["Raw data","Frequency histogram","Descriptive statistics","Empirical probability + fit","Empirical PDF","Candidate PDFs","Empirical + fitted CDFs","Goodness-of-fit comparison","Monte Carlo simulation","Uncertainty percentiles","Full teaching dashboard"]
view=st.radio("Display option",views,horizontal=True)
sp=np.ptp(x) or 1; g=np.linspace(max(0,x.min()-.1*sp),x.max()+.1*sp,700)

if view=="Raw data":
    st.dataframe(pd.DataFrame({"Observation":np.arange(1,len(x)+1),col:x}),use_container_width=True,height=600)
elif view=="Frequency histogram":
    fig,ax=plt.subplots(figsize=(11,6)); ax.hist(x,bins=bins,color=hist_color,edgecolor="black"); ax.set(title="Frequency histogram",xlabel=col,ylabel="Frequency"); fig_show(fig)
elif view=="Descriptive statistics":
    vals=[len(x),x.min(),np.percentile(x,25),np.median(x),x.mean(),np.percentile(x,75),x.max(),stats.iqr(x),x.std(ddof=1),stats.skew(x,bias=False),stats.kurtosis(x,bias=False)]
    labels=["Count","Minimum","Q1","Median","Mean","Q3","Maximum","IQR","Sample SD","Skewness","Excess kurtosis"]
    st.dataframe(pd.DataFrame({"Statistic":labels,"Value":vals}),use_container_width=True,hide_index=True)
elif view=="Empirical probability + fit":
    fig,ax=plt.subplots(figsize=(11,6)); c,e=np.histogram(x,bins=bins); pr=c/len(x); m=(e[:-1]+e[1:])/2
    ax.bar(e[:-1],pr,width=np.diff(e),align="edge",color=hist_color,edgecolor="black",alpha=.65,label="Empirical")
    ax.plot(m,np.diff(fit["dist"].cdf(e,*fit["params"])),"o-",color=line_color,lw=2,label=selected)
    for xx,y,n in zip(m,pr,c):
        if n>0 and y>0: ax.text(xx,y,f"{y*100:.1f}%",ha="center",va="bottom",fontsize=9)
    ax.set(title="Empirical probability histogram + selected fit",xlabel=col,ylabel="Probability within bin"); ax.legend(); fig_show(fig)
elif view=="Empirical PDF":
    fig,ax=plt.subplots(figsize=(11,6)); ax.hist(x,bins=bins,density=True,color=hist_color,edgecolor="black",alpha=.4,label="Empirical density")
    if len(np.unique(x))>1: ax.plot(g,stats.gaussian_kde(x)(g),"--",color="#333333",label="KDE")
    ax.plot(g,fit["dist"].pdf(g,*fit["params"]),color=line_color,lw=2.5,label=f"Selected {selected} PDF")
    ax.set(title="Empirical density and selected PDF",xlabel=col,ylabel="Probability density"); ax.legend(); fig_show(fig)
elif view=="Candidate PDFs":
    fig,ax=plt.subplots(figsize=(11,6)); ax.hist(x,bins=bins,density=True,color=hist_color,edgecolor="black",alpha=.25)
    for z in fits: ax.plot(g,z["dist"].pdf(g,*z["params"]),color=line_color if z is fit else None,lw=2.5 if z is fit else 1.3,label=f"{z['name']} KS={z['ks']:.3f}")
    ax.set(title="Candidate PDFs",xlabel=col,ylabel="Density"); ax.legend(); fig_show(fig)
elif view=="Empirical + fitted CDFs":
    fig,ax=plt.subplots(figsize=(11,6)); s=np.sort(x); ax.step(s,np.arange(1,len(s)+1)/len(s),where="post",label="Empirical CDF")
    for z in fits: ax.plot(g,z["dist"].cdf(g,*z["params"]),color=line_color if z is fit else None,lw=2.5 if z is fit else 1.3,label=z["name"])
    ax.set(title="Empirical and fitted CDFs",xlabel=col,ylabel="P(X ≤ x)"); ax.legend(); fig_show(fig)
elif view=="Goodness-of-fit comparison":
    gof=pd.DataFrame([{"Selected":"Yes" if z is fit else "No","Distribution":z["name"],"KS statistic":z["ks"],"KS p-value":z["pv"],"AIC":z["aic"],"BIC":z["bic"]} for z in fits]).sort_values("KS statistic")
    st.dataframe(gof,use_container_width=True,hide_index=True)
else:
    with st.spinner("Running Monte Carlo simulation..."):
        try: mc=simulate(fit,x,int(nsim)); q=np.percentile(mc,[10,50,90])
        except Exception as e: st.error(str(e)); st.stop()
    if view=="Monte Carlo simulation":
        fig,ax=plt.subplots(figsize=(11,6)); ax.hist(mc,bins=40,density=True,color=hist_color,edgecolor="black",alpha=.65)
        for value,label,color in zip(q,["p10","p50","p90"],["green","orange","red"]): ax.axvline(value,color=color,label=f"{label}={value:.4g}")
        ax.set(title=f"Monte Carlo using {selected}",xlabel=col,ylabel="Simulated density"); ax.legend(); fig_show(fig)
    elif view=="Uncertainty percentiles":
        st.dataframe(pd.DataFrame({"Selected PDF":[selected]*3,"Percentile":["p10","p50","p90"],"Value":q}),use_container_width=True,hide_index=True)
    else:
        fig,aa=plt.subplots(2,2,figsize=(13,9)); c,e=np.histogram(x,bins=bins); pr=c/len(x); m=(e[:-1]+e[1:])/2
        aa[0,0].bar(e[:-1],pr,width=np.diff(e),align="edge",color=hist_color,edgecolor="black",alpha=.65); aa[0,0].plot(m,np.diff(fit["dist"].cdf(e,*fit["params"])),"o-",color=line_color); aa[0,0].set_title("Empirical probability + fit")
        aa[0,1].hist(x,bins=bins,density=True,color=hist_color,edgecolor="black",alpha=.35); aa[0,1].plot(g,fit["dist"].pdf(g,*fit["params"]),color=line_color); aa[0,1].set_title("Selected PDF")
        s=np.sort(x); aa[1,0].step(s,np.arange(1,len(s)+1)/len(s),where="post"); aa[1,0].plot(g,fit["dist"].cdf(g,*fit["params"]),color=line_color); aa[1,0].set_title("CDF")
        aa[1,1].hist(mc,bins=35,density=True,color=hist_color,edgecolor="black",alpha=.65)
        for value,label,color in zip(q,["p10","p50","p90"],["green","orange","red"]): aa[1,1].axvline(value,color=color,label=f"{label}={value:.3g}")
        aa[1,1].legend(); aa[1,1].set_title(f"Monte Carlo: {selected}"); fig_show(fig)

st.caption("Teaching note: select distributions consistent with both goodness-of-fit results and the physical constraints of the variable.")
