import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # noqa 401
import streamlit as st
from io import BytesIO


def mcmc_abtest_from_dist(
    parameter_a: int,
    num_a: int,
    parameter_b: int,
    num_b: int,
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

    return diff_


def plot_abtest(
    parameter_a: int,
    num_a: int,
    parameter_b: int,
    num_b: int,
    day: str,
    grade: str,
    kind: str,
):
    diff_: np.ndarray[np.float64] = mcmc_abtest_from_dist(
        parameter_a, num_a, parameter_b, num_b
    )
    diff_p_ = []
    diff_n_ = []
    for i in range(len(diff_)):
        if diff_[i] >= 0:
            diff_p_.append(diff_[i])
        else:
            diff_n_.append(diff_[i])
    prob_ = len(diff_p_) / len(diff_)
    fig, ax = plt.subplots(figsize=(5, 6))

    sns.histplot(diff_p_, color="red", label=f"Bが高い確率 = {prob_:.4f}", ax=ax)
    sns.histplot(diff_n_, color="blue", label=f"Aが高い確率 = {1-prob_:.4f}", ax=ax)

    ax.vlines(0, 0, 600, colors="gray")
    ax.set_ylim(0, 600)
    ax.set_xlabel(f"{kind}の差")
    ax.set_title(f"{day}・{grade}_{kind}のABテスト")
    ax.legend()
    plt.tight_layout()

    return fig


st.set_page_config(page_title="Baysian_Abtest", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .custom-title {
        font-size: 40px; /* タイトルのフォントサイズ */
        font-weight: bold; /* 太字 */
        color: #FFFFFF; /* 緑色 */
    }
    .custom-header {
        font-size: 30px; /* ヘッダーのフォントサイズ */
        font-weight: normal; /* 標準の太さ */
        color: #2196F3; /* 青色 */
    }
    </style>
    """, unsafe_allow_html=True)

# タイトル
st.markdown('<p class="custom-title">ベイジアンABテストの実行</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    # ヘッダー
    st.markdown('<p class="custom-header">開封数・送信数の入力</p>', unsafe_allow_html=True)
    st.write(
        """
        - 入力は0以上の整数を想定しています。
    """
    )

    a_open = st.number_input("A：開封数", value=100, step=1, format="%d")  # 整数入力
    a_sent = st.number_input("A：送信数", value=500, step=1, format="%d")  # 整数入力
    b_open = st.number_input("B：開封数", value=100, step=1, format="%d")  # 整数入力
    b_sent = st.number_input("B：送信数", value=500, step=1, format="%d")  # 整数入力


with col2:
    st.markdown('<p class="custom-header">ABテスト プロットの出力</p>', unsafe_allow_html=True)

    fig = plot_abtest(a_open, a_sent, b_open, b_sent, "0713", "H2", "開封率")
    st.pyplot(fig)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label="グラフの保存",
        data=buf,
        file_name="plot.png",
        mime="image/png",
    )

st.markdown('<p class="custom-header">ベイジアンABテストの概略</p>', unsafe_allow_html=True)
st.write("""
         パラメータ（ここでは開封率やクリック率）の事前情報と、そのパラメータのもとで生成される実現値（ここでは対象者の開封・クリック行動）
         をもとに、A・Bそれぞれの「事後分布」を得る。この事後分布を比較することで、A・Bの間に差があるかを検証する。
         """)
