import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # noqa 401
import streamlit as st


def mcmc_abtest_from_dist(
    parameter_a: int,
    num_a: int,
    parameter_b: int,
    num_b: int,
    day: str,
    grade: str,
    kind: str,
) -> None:
    with pm.Model() as model:
        theta = pm.Uniform("theta", lower=0.1, upper=0.4, shape=2)
        obs = pm.Binomial(  # noqa F841
            "obs", p=theta, n=[num_a, num_b], observed=[parameter_a, parameter_b]
        )
        trace = pm.sample(5000, chains=2)
        trace_extract = trace.posterior["theta"].data
        thetaa_hat_chain0 = np.array([a[0] for a in trace_extract[0]])
        thetaa_hat_chain1 = np.array([a[0] for a in trace_extract[1]])
        thetab_hat_chain0 = np.array([a[1] for a in trace_extract[0]])
        thetab_hat_chain1 = np.array([a[1] for a in trace_extract[1]])
        thetaa_hat_ = np.concatenate((thetaa_hat_chain0, thetaa_hat_chain1))
        thetab_hat_ = np.concatenate((thetab_hat_chain0, thetab_hat_chain1))
    with model:
        pm.plot_trace(trace, ["theta"], compact=True)
        diff_ = thetab_hat_ - thetaa_hat_

    diff_p_ = []
    diff_n_ = []
    for i in range(len(diff_)):
        if diff_[i] >= 0:
            diff_p_.append(diff_[i])
        else:
            diff_n_.append(diff_[i])
    prob_ = len(diff_p_) / len(diff_)
    g_ = sns.displot(diff_p_, color="red", label=f"Bが高い確率 = {prob_:.4f}")
    g_.map(
        sns.histplot, data=diff_n_, color="blue", label=f"Aが高い確率 = {1-prob_:.4f}"
    )

    plt.vlines(0, 0, 600, colors="gray")
    plt.ylim(0, 600)
    plt.xlabel(f"{kind}の差")
    plt.title(f"{day}・{grade}_{kind}のABテスト")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./out/{day}_{grade}_{kind}.png", dpi=72)
    st.pyplot(plt)


mcmc_abtest_from_dist(121, 659, 102, 672, '0713', '高2', '開封率')
