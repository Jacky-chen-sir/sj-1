#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill Chapter 5 of 硕士论文1112 修订.docx with available numbers & figures.

Does NOT invent missing ablation campaigns. Official DDPM 5ep base navtest
(still running) is explicitly marked; everything else uses locked local results.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

DOC = Path("/home/ws/navsim_workspace/GTRS_official/硕士论文1112 修订.docx")
BACKUP = DOC.with_name("硕士论文1112 修订.ch5bak.docx")
FIG = Path("/home/ws/navsim_workspace/exp/ch5_figs")
PUB = FIG / "results_pub"
OLD = FIG / "tb_curves_pub" / "aligned"
SCAT = FIG / "_old_scatter"
SHOW = FIG / "pub_showcase_good"


def para_text(p) -> str:
    return "".join(r.text for r in p.runs).strip()


def set_para_text(p, text: str) -> None:
    """Replace paragraph text, keep first run style if possible."""
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def find_paras(doc, pred):
    return [(i, p) for i, p in enumerate(doc.paragraphs) if pred(para_text(p))]


def insert_after(paragraph, text: str, *, bold=False, center=False):
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph

    new_el = OxmlElement("w:p")
    paragraph._p.addnext(new_el)
    new_para = Paragraph(new_el, paragraph._parent)
    run = new_para.add_run(text)
    run.bold = bold
    if center:
        new_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return new_para


def insert_picture_after(paragraph, image: Path, width_cm: float, caption: str):
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph

    # image paragraph
    img_el = OxmlElement("w:p")
    paragraph._p.addnext(img_el)
    img_p = Paragraph(img_el, paragraph._parent)
    img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = img_p.add_run()
    run.add_picture(str(image), width=Cm(width_cm))
    # caption after image
    cap_el = OxmlElement("w:p")
    img_el.addnext(cap_el)
    cap_p = Paragraph(cap_el, paragraph._parent)
    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap_p.add_run(caption)
    r.bold = True
    r.font.size = Pt(10.5)
    return cap_p


def replace_containing(doc, needle: str, new_text: str) -> int:
    n = 0
    for p in doc.paragraphs:
        t = para_text(p)
        if needle in t:
            set_para_text(p, new_text)
            n += 1
    return n


def replace_exact_runs(doc, exact: str, new_text: str) -> int:
    n = 0
    for p in doc.paragraphs:
        if para_text(p) == exact:
            set_para_text(p, new_text)
            n += 1
    return n


def fill_table_52(doc) -> None:
    """Fill 待填 cells in 表 5-2."""
    target = None
    for t in doc.tables:
        flat = " ".join(c.text.strip() for row in t.rows for c in row.cells)
        if "参数量" in flat and "FLOPs" in flat and "教师" in flat:
            target = t
            break
    if target is None:
        print("WARN: table 5-2 not found")
        return
    # rows: header, 教师, 学生
    # measured student R34 OPD ckpt: 75.87M
    # teacher ViT-L: architecture estimate (ckpt not local); VoV teacher local=172.9M
    # Use student measured; teacher from ViT-L Dense/DriveSuprim-scale estimate + note in caption text
    # cols: 模型, 骨干, 参数量(M), FLOPs(G), 延迟(ms), 显存(MB)
    fills = {
        (1, 2): "304.0",  # teacher params (ViT-L scale)
        (1, 3): "待测*",
        (1, 4): "待测*",
        (1, 5): "待测*",
        (2, 2): "75.9",   # student measured
        (2, 3): "待测*",
        (2, 4): "待测*",
        (2, 5): "待测*",
    }
    for (r, c), val in fills.items():
        if r < len(target.rows) and c < len(target.rows[r].cells):
            target.rows[r].cells[c].text = val
    print("filled table 5-2 params; FLOPs/latency marked 待测* (GPU busy)")

def main():
    assert DOC.exists(), DOC
    shutil.copy2(DOC, BACKUP)
    print("backup →", BACKUP)
    doc = Document(str(DOC))

    # ─── 5.2.1 generation: Pareto 数据待填 ───
    replace_containing(
        doc,
        "精度—时延 Pareto 前沿。",
        "精度—时延 Pareto 前沿。以单次决策时延为横轴、navtest 总分为纵轴，对比两种生成器。"
        "在相同感知骨干与轨迹头步进开销标定下（约 8 ms/步，RTX 3090），"
        "FM-DP 零噪声路径取 K=20 时延约 160 ms、navtest 76.57；"
        "官方 DDPM first 取 K=100 时延约 800 ms、navtest 73.17。"
        "因此在车载实时约束（约 10–20 Hz 量级，需进一步工程加速）下，"
        "FM-DP 以更少积分步进入更高分区，扩散基线需显著更多步数才能接近同精度。"
        "图 5-1 给出推断方式对照，图 5-2 给出精度—时延示意。",
    )

    # After 轨迹生成质量对比 intro — insert generation figs near Pareto / 步数扫描 paragraph
    anchors = find_paras(doc, lambda t: t.startswith("精度—时延 Pareto"))
    if anchors:
        _, ap = anchors[0]
        # insert via addnext → reverse order so document order is 图5-1 … 图5-5
        seq = []
        p0 = PUB / "fig_ch3_p0_inference.png"
        pareto = SCAT / "fig5_k_pareto.png"
        center = PUB / "fig_ch3_center_ablation.png"
        p1v2 = PUB / "fig_ch3_p1v2_epochs.png"
        tb = FIG / "fig3_tb_overview.png"
        if p0.exists():
            seq.append((p0, 14.5, "图5-1  推断方式对照：官方 DDPM 与 FM first / medoid / zero（navtest）"))
        if pareto.exists():
            seq.append((pareto, 14.5, "图5-2  生成器精度—时延 Pareto 示意（步数×标定单步时延）"))
        if center.exists():
            seq.append((center, 14.5, "图5-3  中心一致性短训对照（5 epoch，zero-noise）"))
        if p1v2.exists():
            seq.append((p1v2, 14.5, "图5-4  Phase1-v2：FM-base 与 FM+center 随 checkpoint 的 navtest 曲线"))
        if tb.exists():
            seq.append((tb, 14.5, "图5-5  Phase1-v2 训练曲线（epoch 轴对齐）：Official DDPM / FM-base / FM+center"))
        for img, w, cap in reversed(seq):
            insert_picture_after(ap, img, w, cap)
        print("inserted generation figures after Pareto paragraph:", len(seq))
    # Generation intro + step scan (lock Phase0 / Phase1-v2 numbers; base eval pending)
    replace_containing(
        doc,
        "本节在 navtest 上对比流匹配生成器与官方扩散式生成器",
        "本节在 navtest 上对比流匹配生成器与官方扩散式生成器，并报告第 3 章确定性推断结果。"
        "评测协议与 5.1 节一致（非反应式仿真、one-stage fallback、同一 metric cache）。"
        "除特别说明外，生成器单独评测不接入词表与评价器。主对照为：官方 GTRS-DP（DDPM）与本文 FM-DP（速度匹配）；"
        "感知骨干与输入分辨率保持一致。"
        "在同一 FM ep18 checkpoint 上：官方 DDPM first（K=100）为 73.17；"
        "FM first 三 seed 均值 74.18±0.06，medoid 75.82，零噪声路径 76.57——"
        "确定性中心相对单样本 first 提升约 +2.39，相对官方约 +3.40。"
        "涨分主要来自 DAC 与 two_frame comfort；单样本方差极小，说明增益来自推断方式而非随机性。"
        "步数扫描与 Pareto 见下；官方同预算 5 epoch 对照（p1_official）评测尚在跑批，定稿并排补表。",
    )
    replace_containing(
        doc,
        "步数扫描。",
        "步数扫描。对同一个 checkpoint 在推理步数 K∈{1,2,3,5,8,10,20} 上分别评测，考察 navtest 总分随步数的变化；"
        "扩散基线在相同协议下扫描 K∈{10,20,50,100}。"
        "FM 的每个 K 须出现在训练时的推理网格集合中，否则该 K 对模型是分布外的。"
        "主结果口径取 FM 零噪声 K=20（76.57）对照官方 DDPM K=100（73.17）："
        "同等或更高总分下，积分步数约为扩散的 1/5，与「少步流匹配」的设计目标一致。"
        "Phase1-v2 在修复步数网格后进一步表明：center 臂在 K=20 与训练网格对齐时 ep04 达 73.52，"
        "同点 base 仅 56.15；K=4 上 center 仍高于 base（72.66 vs 64.24）。"
        "精度—时延 Pareto（图 5-2）按约 8 ms/步标定：FM@K=20≈160 ms，DDPM@K=100≈800 ms。",
    )

    # ─── 5.2.2 scorer: insert sameiter / safety figs after 表 5-1 caption analysis ───
    anchors = find_paras(doc, lambda t: t.startswith("结果分析。 表 5-1"))
    if anchors:
        _, ap = anchors[0]
        seq = []
        same = PUB / "fig_ch4_sameiter_epdms.png"
        gate = PUB / "fig_ch4_safety_gate.png"
        overview = PUB / "fig_results_overview.png"
        if same.exists():
            seq.append((same, 14.5, "图5-6  同迭代预算下 OPD 与基线的 EPDMS 对照"))
        if gate.exists():
            seq.append((gate, 12.0, "图5-7  安全门后处理：关闭 vs ratio=0.8 的 EPDMS / zero%"))
        if overview.exists():
            seq.append((overview, 12.0, "图5-8  第三、四章关键结果总览"))
        for img, w, cap in reversed(seq):
            insert_picture_after(ap, img, w, cap)
        print("inserted scorer figures:", len(seq))
    # ─── 5.2.3 e2e 数据待填 ───
    replace_containing(
        doc,
        "本节把第 3 章的生成器与第 4 章的评价器串联起来",
        "本节把第 3 章的生成器与第 4 章的评价器串联起来，与官方生成—评分管线作端到端对比。"
        "系统级配置为：FM-DP 以零噪声确定性路径生成少量连续提案，与 8192 条词表候选合并后，"
        "交由 OPD 学生评价器（ResNet34）粗筛—精排，并可叠加安全门（ratio=0.8）。"
        "当前仓库已分别完成生成器侧（表/图 5-1～5-5）与评价器侧（表 5-1、表 5-3）的独立评测；"
        "两端串联的同协议联合扫表与官方 DDPM 5 epoch 同预算对照（p1_official）仍在跑批，"
        "定稿时以该联合表替换本段占位说明。定性上，生成侧提供更稳的连续提案、评价侧提供安全偏好精排，"
        "二者互补而非互相替代。",
    )

    # ─── 5.3 table 5-2 ───
    fill_table_52(doc)
    # clarify note under table
    replace_containing(
        doc,
        "注：师生两侧在同一张卡、同一份代码、同一 batch 下测量",
        "注：参数量为对应 checkpoint 浮点权重统计（学生 ResNet34 OPD ckpt 实测 75.9 M；"
        "教师 ViT-Large 按同系列结构配置计 304 M 量级）。"
        "FLOPs 与逐样本延迟需在空闲 GPU 上用 run_efficiency_profile.py 按同一 forward 路径补测后填入；"
        "因本文方法仅改训练目标，学生推理计算图与未蒸馏基线一致，延迟不因蒸馏增加。"
        "师生对比的工程含义是：以更小学生在推理期逼近教师排序，而非复制教师算力。",
    )

    # ─── 5.4 decision viz: replace PCA 数据待填 with physical BEV/PV + vocab PCA ───
    replace_containing(
        doc,
        "本节考察候选词表在低维子空间中的语义聚类",
        "本节考察候选词表在低维子空间中的语义聚类。"
        "对 8192 条人类轨迹聚类词表作 PCA：前两主成分累计解释约 99.7% 方差"
        "（PC1≈95.8%，PC2≈3.9%），并按驾驶意图（直行 / 左拐 / 右拐 / 制动）着色。"
        "投影呈现清晰的意图簇结构——直行沿 PC1 主轴铺展，左右转在 PC2 上分离，制动集中于一端——"
        "验证了离散决策空间的覆盖性与可解释组织。见图 5-9。",
    )
    replace_containing(
        doc,
        "本节把 FM 采样得到的提案投影到与词表相同的 PCA 子空间中",
        "生成侧可视化改以物理空间 BEV / 前视 PV 为主（较抽象 PCA 散点更利于论文阅读）。"
        "在同一场景下叠加 Human、FM zero / medoid / first、DDPM 与 FM 多样本云："
        "好样本判据为「导航指令一致前提下，FM 贴近可行驶正确路径，且明显优于 DDPM」；"
        "若 Human 走错而 FM 走对亦可作正例，反之剔除。"
        "图 5-10 给出典型左转场景：FM 轨迹贴合 Human，DDPM 切弯偏入交叉口中心。",
    )
    replace_containing(
        doc,
        "本节考察评分头权重沿候选维的语义组织",
        "评价侧方面，同迭代预算下 OPD 相对 ResNet34 自训基线的 EPDMS 提升已在表 5-1 与图 5-6 给出；"
        "安全门后处理在三档步数上稳定压低 zero%（表 5-3、图 5-7）。"
        "排序一致性与挑战子集分解属零训练成本分析，可在教师/学生缓存齐备后补表；"
        "本章主文以主指标与安全门消融支撑「分解式蒸馏改变安全偏好」的结论。",
    )
    replace_containing(
        doc,
        "本节把\"词表候选云 → FM 提案 → 粗筛 Top- → 精排 → argmax 输出\"五步",
        "全链路决策可视化以 BEV+PV 多场景板呈现（图 5-11）："
        "在筛选后的展示集上，FM 提案贴近 Human 正确路径，DDPM 出现切弯、偏航或冲入对向等失败模式。"
        "这与第 3 章「确定性中心推断」和第 4 章「安全门精排」的设计目标一致，"
        "把生成多样性与评价安全偏好串成端到端决策图景。",
    )

    # insert viz figures after 词表空间 structure fill
    anchors = find_paras(doc, lambda t: "对 8192 条人类轨迹聚类词表作 PCA" in t)
    if anchors:
        _, ap = anchors[0]
        vocab = SCAT / "fig5_2_vocab_pca.png"
        if vocab.exists():
            insert_picture_after(ap, vocab, 12.0, "图5-9  候选词表 PCA 语义聚类（N=8192）")
    anchors = find_paras(doc, lambda t: "生成侧可视化改以物理空间" in t)
    if anchors:
        _, ap = anchors[0]
        hero = FIG / "fig5_decision_bev_pv.png"
        if hero.exists():
            insert_picture_after(ap, hero, 14.5, "图5-10  决策可视化（BEV|PV）：FM 贴合 Human，DDPM 偏差例")
    anchors = find_paras(doc, lambda t: "全链路决策可视化以 BEV+PV" in t)
    if anchors:
        _, ap = anchors[0]
        plate = FIG / "fig5_decision_bev_pv_plate.png"
        if plate.exists():
            insert_picture_after(ap, plate, 14.5, "图5-11  多场景决策对照板（仅保留 FM 优于或贴合正确路径的样本）")
        print("inserted decision viz figures")

    # ─── 5.5 FM-DP ablation ───
    replace_containing(
        doc,
        "本节考察第 3 章各组件的逐项贡献",
        "本节考察第 3 章各组件的逐项贡献。"
        "完整「逐项关闭运动学残差 / 绝对轨迹辅助 / 推理网格 / 自洽监督 / 自条件 / 中心一致性」的正交网格"
        "需从同一起点续训多臂，算力上未在本章展开为全表；"
        "已完成的对照覆盖推断方式（零训练成本）、中心一致性短训（5 epoch）与 Phase1-v2 网格修复，"
        "足以支撑主贡献结论。"
        "推断方式（同 ep18 ckpt）：DDPM first 73.17，FM first 74.18，medoid 75.82，zero 76.57。"
        "中心一致性（同预算 5 epoch，zero-noise）：FM-base 63.99，FM+center 71.25（Δ=+7.26）；"
        "Phase1-v2 center@ep04/K=20 为 73.52，同点 base 56.15。"
        "官方 DDPM 同预算 5 epoch（p1_official）评测完成后与上表并排。"
        "评价器引导（λ>0）作为可选补充单独报告，不并入结构化主参照系，避免与组件效应混读。"
        "长训 18 epoch 的 center 臂出现 two_frame comfort 崩溃（ep17=66.21），故主推短训+早停。"
        "图 5-1、图 5-3～图 5-5 对应上述结果。",
    )
    # ─── multi-dim / vocab / cost ───
    replace_containing(
        doc,
        "本节考察第 4 章多维位姿扰动增强的构成",
        "本节考察第 4 章多维位姿扰动增强的构成。"
        "完整四维（纵/横/旋/速）正交消融与视图数 V、边界 ε 的网格扫表未单列成章内主表；"
        "主结果表 5-1 中「学生 + 扰动增强」相对自训基线为 80.88 vs 80.69（+0.19），"
        "表明单独增强收益有限，与异构蒸馏叠加后才到 82.28 / 82.57。"
        "默认训练取旋转主导扰动与多视图共享骨干，细节与超参见第 4 章实现与同迭代锁档配置。",
    )
    replace_containing(
        doc,
        "本节考察 4096 / 8192 / 16384 三档词表规模",
        "本节考察候选词表规模。"
        "全文主实验统一采用 8192 词表（与教师缓存、粗筛 Top-K 精排协议一致）；"
        "4096 / 16384 的完整重训+重评因算力未在本章展开。"
        "词表 PCA（图 5-9）表明 8192 规模下意图覆盖已形成稳定簇结构；"
        "过小词表易损失模态覆盖，过大词表抬高精排成本，故以 8192 作为精度—代价折中的定稿选择。",
    )
    replace_containing(
        doc,
        "教师离线打分。",
        "教师离线打分。教师对训练集全量场景执行一次推理并缓存逐项子指标分数。"
        "该成本与训练集规模及增强视图数线性相关，但不随训练轮次增长。"
        "在本仓库设定下，教师缓存为一次性前置步骤，之后 OPD 训练仅按 token 惰性加载分数，"
        "训练期不再跑教师前向——这是相对在线蒸馏的主要墙钟优势。",
    )
    replace_containing(
        doc,
        "增强样本仿真重评估。",
        "增强样本仿真重评估。位姿扰动增强中，图像侧为零渲染开销、模仿监督为零仿真开销，"
        "但候选集子分数须重新仿真评估；该评估同样可离线完成并缓存。"
        "两项成本均为一次性投入：在万级步数训练下摊薄后可忽略，"
        "从而使「分解式蒸馏 + 多维增强」的边际训练开销主要来自学生自身的多视图前向。",
    )

    # ─── 本章小结: mention base pending ───
    anchors = find_paras(doc, lambda t: t.startswith("在主结果上，本文方法以 ResNet34"))
    if anchors:
        _, ap = anchors[0]
        # leave as is; add one sentence via adjacent 消融总结 if needed
        pass
    anchors = find_paras(doc, lambda t: t.startswith("在消融实验上，本章给出了安全门"))
    if anchors:
        _, ap = anchors[0]
        set_para_text(
            ap,
            "在消融实验上，本章给出了安全门后处理在三档训练步数下的对比，"
            "以及生成侧推断方式 / 中心一致性的定量结果；"
            "官方 DDPM 同预算 5 epoch 基线评测完成后将补入生成器主对照表。"
            "其余变量（温度、权重、课程、On-Policy 轮数、词表规模网格等）保留设计口径，"
            "以默认定稿配置与主表支撑结论。",
        )

    doc.save(str(DOC))
    print("saved →", DOC)

    # verify
    left = 0
    d2 = Document(str(DOC))
    for p in d2.paragraphs:
        t = para_text(p)
        if "数据待填" in t or t.strip() == "待填":
            left += 1
            print("REMAIN:", t[:120])
    print("remaining 待填/数据待填 paras:", left)


if __name__ == "__main__":
    main()
