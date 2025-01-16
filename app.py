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
    lower: float,
    upper: float,
) -> None:

    with pm.Model() as model:
        theta = pm.Uniform("theta", lower=lower, upper=upper, shape=2)
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
    lower: float,
    upper: float,
):
    diff_: np.ndarray[np.float64] = mcmc_abtest_from_dist(
        parameter_a, num_a, parameter_b, num_b, lower, upper
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

    return fig, prob_


st.set_page_config(page_title="Baysian_Abtest", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .custom-title {
        font-size: 60px; /* タイトルのフォントサイズ */
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

lower = st.number_input(
    label='開封率の下限',
    min_value=0.0,        # 最小値
    max_value=1.0,
    value=0.1,            # 初期値
    step=0.1,             # 増減ステップ
    format="%.3f")
upper = st.number_input(
    label='開封率の上限',
    min_value=0.0,        # 最小値
    max_value=1.0,
    value=0.4,            # 初期値
    step=0.1,             # 増減ステップ
    format="%.3f")

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

    fig, prob = plot_abtest(a_open, a_sent, b_open, b_sent, "0713", "H2", "開封率", lower, upper)
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

st.markdown('<p class="custom-header">確率</p>', unsafe_allow_html=True)
st.write(f"""
         Aが高い確率… {1-prob:.4f} <br>
         Bが高い確率… {prob:.4f} <br>
         """, unsafe_allow_html=True)

st.markdown('<p class="custom-header">上記プロットの見方・使い方</p>', unsafe_allow_html=True)
st.markdown("""
        まず、これまでの経験から、開封率の下限と上限を設定する（事前分布の設定）。<br>
         上記プロットは、Aの開封率（クリック率）とBの開封率（クリック率）の差の確率分布である。
         - 正の領域（右側・赤色の部分）の面積は、BがAより大きくなる確率となる。
         - 負の領域（左側・青色の部分）の面積は、AがBより大きくなる確率となる。
         ある閾値をあらかじめ決めておき（ex. 0.95（95%）・0.9（90%））、それよりも大きい確率が出力された場合、
         AとBの間に統計的に有意な差があると判断する。
         """, unsafe_allow_html=True)


st.markdown('<p class="custom-header">ベイジアンABテストの概略</p>', unsafe_allow_html=True)
st.markdown("""
         パラメータ（ここでは開封率やクリック率）の事前分布<sup>※1</sup>と、そのパラメータのもとで生成される
         実現値（ここでは対象者の開封・クリック行動）をもとに、A・Bそれぞれの事後分布<sup>※2</sup>を得る。
         この事後分布を比較することで、A・Bの間に差があるかを検証する。
         上記プロットは、MCMCサンプリングと呼ばれる方法で生成された乱数を図示したもので、
         このサンプリングは、事後分布からのサンプリングとみなせる。すなわち、上記プロットは
         自らが設定した事前分布と観測値の情報から情報を更新し、A・Bを生成したであろう分布を求め、
         その分布を比較することで、効果を検証する。 <br>
         <p class="indent-after-first-line">
         <sup>※1</sup>事前分布…「まだ観測していないけれど持っている予想や信念」を確率として表したもの。
         この分布を使い、新しいデータを取り入れて、予想を更新していく
         <sup>※2</sup>事後分布…新しい情報を取り入れた後に更新された、予想や信念を確率で表したもの。
         </p>
         """, unsafe_allow_html=True)
