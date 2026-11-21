#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite Chapter 5 of 硕士论文1112 修订.docx in reference-thesis style.

Structure (aligned with 高胤峰 / 尤国良 experiment chapters):
  实验设置 → 基线方法 → 定量结果与分析 → 消融 → 可视化 → 局限 → 小结

Keeps locked numbers/tables/figures; removes unfinished-ablation wishlists;
leads each result subsection with the verdict, then table, then contrast.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

DOC = Path("/home/ws/navsim_workspace/GTRS_official/硕士论文1112 修订.docx")
BACKUP = DOC.with_name("硕士论文1112 修订.ch5rewrite_bak.docx")


def para_text(p) -> str:
    return "".join(r.text for r in p.runs).strip()


def set_run_font(run, *, size=12, bold=False, east_asia="宋体"):
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), east_asia)


def make_paragraph(text: str, *, style="Normal", bold=False, center=False, size=12):
    p = OxmlElement("w:p")
    # style
    pPr = OxmlElement("w:pPr")
    pStyle = OxmlElement("w:pStyle")
    # Map logical roles to styles present in this thesis
    style_map = {
        "Normal": "Normal",
        "Heading": "List Paragraph",  # thesis uses List Paragraph for section titles
        "Caption": "Normal",
    }
    pStyle.set(qn("w:val"), style_map.get(style, "Normal"))
    pPr.append(pStyle)
    if center or style == "Caption":
        jc = OxmlElement("w:jc")
        jc.set(qn("w:val"), "center")
        pPr.append(jc)
    # first-line indent for body
    if style == "Normal" and text and not center:
        ind = OxmlElement("w:ind")
        ind.set(qn("w:firstLine"), "480")  # ~2 chars
        pPr.append(ind)
    p.append(pPr)

    if text:
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        if bold or style == "Heading":
            b = OxmlElement("w:b")
            rPr.append(b)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int((14 if style == "Heading" else size) * 2)))
        rPr.append(sz)
        szCs = OxmlElement("w:szCs")
        szCs.set(qn("w:val"), str(int((14 if style == "Heading" else size) * 2)))
        rPr.append(szCs)
        rfonts = OxmlElement("w:rFonts")
        east = "黑体" if style == "Heading" else "宋体"
        rfonts.set(qn("w:ascii"), "Times New Roman")
        rfonts.set(qn("w:hAnsi"), "Times New Roman")
        rfonts.set(qn("w:eastAsia"), east)
        rPr.append(rfonts)
        r.append(rPr)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        p.append(r)
    return p


def clone(el):
    return copy.deepcopy(el)


def find_ch5_range(doc: Document):
    start = end = None
    for i, p in enumerate(doc.paragraphs):
        t = para_text(p)
        if t == "实验结果与决策过程分析":
            start = p._p
        if start is not None and t == "参考文献" and i > 540:
            end = p._p
            break
    if start is None or end is None:
        raise RuntimeError(f"Cannot locate Ch5 range: start={start} end={end}")
    return start, end


def detach_blocks(doc: Document, start, end):
    """Remove Ch5 children; return dict of reusable tables and figure blocks."""
    body = doc.element.body
    paras = {p._p: p for p in doc.paragraphs}
    tables = {t._tbl: (i, t) for i, t in enumerate(doc.tables)}

    table_by_header = {}
    fig_blocks = {}  # caption_key -> (img_el, cap_el)
    other_keep = []

    children = list(body.iterchildren())
    in_range = False
    pending_img = None

    for child in children:
        if child is start:
            in_range = True
        if child is end:
            break
        if not in_range:
            continue

        tag = child.tag.split("}")[-1]
        if tag == "tbl":
            idx, tbl = tables[child]
            header = " | ".join(
                c.text.strip().replace("\n", " ")[:30] for c in tbl.rows[0].cells
            )
            table_by_header[header] = clone(child)
            body.remove(child)
            pending_img = None
        elif tag == "p":
            has_draw = bool(child.findall(".//" + qn("w:drawing")))
            txt = para_text(paras[child]) if child in paras else ""
            if has_draw:
                pending_img = child
                body.remove(child)
            elif pending_img is not None and txt.startswith("图"):
                # normalize key: 图5-1 / 图5-10 etc.
                key = txt.split()[0].replace(" ", "")
                fig_blocks[key] = (clone(pending_img), clone(child))
                body.remove(child)
                pending_img = None
            else:
                # drop text paragraphs; if orphan pending_img without caption, keep keyed by empty
                if pending_img is not None:
                    # image without following caption — stash temporarily
                    other_keep.append(("orphan_img", clone(pending_img)))
                    pending_img = None
                body.remove(child)
        else:
            body.remove(child)

    return table_by_header, fig_blocks


def insert_before(ref_el, new_el):
    ref_el.addprevious(new_el)


def append_sequence(end_el, elements):
    """Insert elements in order just before end_el (参考文献)."""
    for el in elements:
        end_el.addprevious(el)


def caption_para(text: str):
    return make_paragraph(text, style="Caption", bold=True, center=True, size=10.5)


def heading(text: str):
    return make_paragraph(text, style="Heading", bold=True)


def body(text: str):
    return make_paragraph(text, style="Normal")


def note(text: str):
    return make_paragraph(text, style="Normal", size=10.5)


def fig_pair(fig_blocks, key, new_caption: str | None = None):
    """Return [img, caption] elements; update caption text if provided."""
    if key not in fig_blocks:
        # try loose match
        for k in fig_blocks:
            if k.startswith(key) or key.startswith(k):
                key = k
                break
    if key not in fig_blocks:
        print(f"[warn] missing figure {key}, available={list(fig_blocks)}")
        return [caption_para(new_caption or key)]
    img, cap = fig_blocks[key]
    img = clone(img)
    cap = clone(cap)
    if new_caption:
        # replace all w:t in caption
        texts = cap.findall(".//" + qn("w:t"))
        if texts:
            texts[0].text = new_caption
            for t in texts[1:]:
                t.text = ""
    return [img, cap]


def table_el(table_by_header, *needles):
    for h, el in table_by_header.items():
        if all(n in h for n in needles):
            return clone(el)
    print(f"[warn] missing table needles={needles} headers={list(table_by_header)}")
    return make_paragraph(f"[缺表: {needles}]", style="Normal")


def build_ch5(table_by_header, fig_blocks):
    """Return ordered list of XML elements for the rewritten chapter."""
    T_infer = table_el(table_by_header, "推断方式", "navtest")
    T_main = table_el(table_by_header, "方法", "EPDMS")
    T_eff = table_el(table_by_header, "参数量", "FLOPs")
    T_budget = table_el(table_by_header, "预算", "评测设定")
    T_gate = table_el(table_by_header, "安全门", "zero%")

    E = []

    # ---- title ----
    E.append(heading("实验结果与决策过程分析"))
    E.append(
        body(
            "本章在 NAVSIM 基准上对第 3 章流匹配轨迹生成器（FM-DP）与第 4 章分解式异构蒸馏评价器（OPD）"
            "进行系统实验验证。叙述顺序参照完整实验章节的通行写法：先固定实验配置与对比口径，再列出基线方法，"
            "随后给出主结果的定量对照与分析，再通过消融实验剥离各组件贡献，最后辅以决策过程可视化，并在章末归纳结论。"
            "除特别说明外，全部数字均来自同一评测管线与锁档结果，可复核。"
        )
    )

    # ============================================================
    # 5.1 实验设置
    # ============================================================
    E.append(heading("实验设置"))
    E.append(heading("数据集与评测协议"))
    E.append(
        body(
            "全部实验在 NAVSIM 基准上进行。该基准基于 nuScenes 构建开环规划评测，训练集 navtrain 与测试集 navtest "
            "按场景日志划分、分布不重叠。评测采用非反应式仿真：规划器对每个场景输出一次轨迹，仿真器依据交通规则与运动学约束打分。"
            "主报告口径统一为 navtest、one-stage fallback，场景规模约 1.2×10⁴；生成器单独评测时不接入词表与评价器，评价器评测时采用默认 8192 词表与 Top-K=256 粗筛—精排协议。"
        )
    )

    E.append(heading("评价指标"))
    E.append(
        body(
            "主指标为 EPDMS（Extended PDMS），由九项子指标组成。安全组（无责任碰撞 NC、可行驶区域合规 DAC、"
            "行驶方向合规 DDC、交通灯合规 TLC）采用乘性聚合：任一项失效则该场景得分为零；舒适组（碰撞时间 TTC、"
            "通行效率 EP、车道保持 LK、历史舒适度 HC、扩展舒适度 EC）采用加权加性聚合，权重分别为 5、5、2、2、2。"
            "需要区分“评测口径”与“网络口径”：评测使用全部九项，而网络侧对其中八项设置预测头——EC 由跨相邻两帧的解析量在聚合阶段计算，不是逐帧可学习的量。"
            "此外，本章同步报告安全违规场景占比 zero%：该量直接反映乘性安全项归零的频率，是安全门后处理的核心观测指标。"
            "除特别注明外，表格中分数均以百分制给出。"
        )
    )

    E.append(heading("实现细节与训练配置"))
    E.append(
        body(
            "模型配置方面，轨迹评价器采用师生异构结构：教师为 ViT-Large 骨干、输入分辨率 1024×256；学生为 ResNet34 骨干、"
            "输入分辨率 512×2048。候选词表规模默认 N=8192，精排阶段保留 Top-K=256。轨迹生成器采用 VoV-99 骨干与 5 层 "
            "Transformer 解码器；流匹配主结果积分步数 K=20，官方扩散对照步数 K=100；边界修正与确定性推断规则与第 3 章一致。"
        )
    )
    E.append(
        body(
            "训练配置方面，全部训练与评测锁定 fp32，在 3×RTX 3090 上完成。优化器为 AdamW，学习率采用余弦退火。"
            "评价器 OPD 主实验训练至 13000 step，并在 5000 / 9000 step 报告中途点；生成器中心一致性短训预算为 5 epoch，"
            "长训对照为 18 epoch。教师对训练集全量场景离线打分并缓存，训练期仅按场景 token 惰性加载分数，不参与在线前向。"
            "效率测量在单卡 RTX 3090（GPU0）、batch=1、同一 agent.forward 路径下完成，延迟取中位数（p50）。"
        )
    )
    E.append(
        body(
            "数值精度说明：混合精度下，EPDMS 乘性聚合涉及连乘与对数运算，低精度下溢会使安全因子乘积提前归零，从而污染训练信号。"
            "因此本文不以“节省显存”为由启用混合精度，而以 fp32 保证乘性安全监督的数值稳定。"
        )
    )

    E.append(heading("公平对比约定"))
    E.append(
        body(
            "为保证结论可解释，本章统一如下对比约定。（1）同一评测管线：所有对比方法使用同一份 metric cache、同一阶段一策略与同一聚合脚本；"
            "跨方法比较仅在口径一致时进行。（2）同构比较优先：评价器消融以自复现 ResNet34 自训基线（EPDMS=80.69）为参照，"
            "不以文献中不同管线报告的 83.1 作为差值基准；生成器同预算对照均自同一官方 checkpoint 重初始化轨迹头后短训。"
            "（3）同变量原则：安全门实验仅改 safety_gate_ratio（0 与 0.8），其余超参、权重与 checkpoint 不变。"
            "（4）跨帧聚合：若映射缺失导致退化为不含 EC 的八项总分，则与九项口径不可比，正文不予混报。"
        )
    )

    # ============================================================
    # 5.2 基线方法
    # ============================================================
    E.append(heading("基线方法"))
    E.append(
        body(
            "为分别检验生成器与评价器的贡献，本章设置如下基线与参照。"
        )
    )
    E.append(
        body(
            "（1）生成器基线：官方 GTRS-DP 扩散配方（DDPM）。成熟 checkpoint 上以 first 样本、K=100 报告；"
            "同预算 5 epoch 对照以 ep04、K=100 报告。本文方法对应 FM-DP：速度匹配训练，主推零噪声确定性路径（K=20），"
            "并报告 first / medoid 作为推断方式对照。"
        )
    )
    E.append(
        body(
            "（2）评价器基线：DriveSuprim 风格的 ResNet34 自训同构基线（本文复现，EPDMS=80.69）；"
            "同迭代预算下的官方 GTRS ResNet34 配方作为扰动增强对照；ViT-Large 教师分数（87.1）作为能力上界，"
            "用于衡量蒸馏后的师生差距，而非要求学生超越教师。"
        )
    )
    E.append(
        body(
            "（3）消融内部对照：生成器侧比较 FM-base 与 FM+center（是否加入中心一致性监督），以及长训与短训+早停；"
            "评价器侧比较“自训 → +扰动增强 → +异构蒸馏 → +安全门”的四级拆解，以及安全门关闭（ratio=0）与开启（ratio=0.8）。"
        )
    )

    # ============================================================
    # 5.3 定量结果
    # ============================================================
    E.append(heading("定量实验结果与分析"))

    # ---- 5.3.1 evaluator ----
    E.append(heading("轨迹评价器主结果"))
    E.append(
        body(
            "评价器侧的核心结论是：在不改变学生推理结构的前提下，分解式异构蒸馏把 ResNet34 自训基线从 80.69 提升至 82.57"
            "（Δ=+1.88），并缩小与 ViT-Large 教师（87.1）的差距至 4.53 分。表 5-1 给出 navtest 上的 EPDMS 与九项子指标对照。"
        )
    )
    E.append(T_main)
    E.append(caption_para("表 5-1  navtest 评价器主结果对比"))
    E.append(
        note(
            "注：文献中同类 ResNet34 自训基线曾报告 83.1，因训练与评测管线不同，本文一律以自复现 80.69 为比较基准。"
            "「学生 + 异构蒸馏」对应同迭代预算 13000 step 的 OPD；「本文方法（完整）」在此基础上叠加精排安全门（ratio=0.8）。"
            "教师行仅给出总分上界，子指标从略。"
        )
    )
    E.append(
        body(
            "由表 5-1 可见，四级拆解的增益结构十分清晰。相对自训基线，仅加扰动增强得到 80.88（+0.19），说明增强本身不是主因；"
            "再叠加异构蒸馏升至 82.28（相对增强组 +1.40，相对基线 +1.59），构成主要涨分；完整方法（含安全门）达到 82.57。"
            "因此，分解式异构蒸馏是评价器性能提升的主导因素，安全门提供额外的安全偏好精排增益。"
        )
    )
    E.append(
        body(
            "子指标层面，完整方法相对自训基线呈现明确的安全侧偏好：NC、DAC、DDC、TLC 四项安全指标全部上升"
            "（其中 DAC +1.96、NC +0.71），TTC 与 EC 同步改善；通行效率 EP 由 84.91 降至 82.18。"
            "这一涨跌组合与第 4 章设计一致——安全门在精排时排斥安全乘积偏低的候选，用部分舒适度换取更少的整场景零分。"
            "由于 EPDMS 对安全项采用乘性合取，降低 zero% 对总分是净收益。同迭代预算曲线见图 5-6，安全门开关对照见图 5-7，关键结果总览见图 5-8。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-6", "图5-6  同迭代预算下 OPD 与基线的 EPDMS 对照"))
    E.extend(fig_pair(fig_blocks, "图5-7", "图5-7  安全门后处理：关闭 vs ratio=0.8 的 EPDMS / zero%"))
    E.extend(fig_pair(fig_blocks, "图5-8", "图5-8  第三、四章关键结果总览"))

    # ---- 5.3.2 efficiency ----
    E.append(heading("推理效率分析"))
    E.append(
        body(
            "评价器改进全部发生在训练阶段，推理计算图与未蒸馏学生一致。表 5-2 给出师生在 RTX 3090 上的参数量、FLOPs、"
            "延迟与峰值显存，用以支撑“轻量部署、接近教师排序能力”的工程结论。"
        )
    )
    E.append(T_eff)
    E.append(caption_para("表 5-2  师生推理效率对比（RTX 3090，batch=1）"))
    E.append(
        note(
            "注：参数量为推理槽位浮点权重；FLOPs 由 torch.utils.flop_counter 统计主要算子（G/样本）；"
            "延迟为 CUDA event 的每样本中位数 p50；显存为单次前向峰值 max_memory_allocated。"
            "OPD 训练 checkpoint 的全量 state_dict 另含 EMA 等，不等于部署参数量。"
        )
    )
    E.append(
        body(
            "相对教师（321.3 M / 812.0 G / 154.6 ms / 2564 MB），学生为 37.9 M / 364.2 G / 101.0 ms / 477 MB："
            "参数量约 1/8.5，峰值显存约 1/5.4，延迟约 2/3，FLOPs 约为教师的 45%。结合表 5-1，学生在约 1/8 参数量下"
            "将自训基线提升 1.88 分并逼近教师排序能力，说明蒸馏把教师的安全偏好写入学生权重，却不把教师的推理成本带上车。"
            "效率柱图见图 5-12。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-12", "图5-12  师生推理效率对比（参数量 / FLOPs / 延迟 / 显存）"))
    E.append(
        body(
            "训练成本方面，教师离线打分与增强样本仿真重评估均为一次性前置开销，不随训练步数增长；OPD 训练期只加载缓存分数。"
            "相对在线蒸馏“每步跑教师”的线性成本，这一结构在万级步数训练下摊薄后可忽略，是离线分解式蒸馏的工程优势。"
        )
    )

    # ---- 5.3.3 generator ----
    E.append(heading("轨迹生成器主结果"))
    E.append(
        body(
            "生成器侧的核心结论有三：其一，在同一成熟 FM checkpoint（ep18）上，确定性零噪声路径优于随机 first 样本，"
            "并优于官方 DDPM first；其二，在同预算 5 epoch 短训下，FM+center 显著高于官方 DDPM 与 FM-base；"
            "其三，达到更高分数所需积分步数约为扩散基线的 1/5。表 5-3 给出推断方式对照。"
        )
    )
    E.append(T_infer)
    E.append(caption_para("表 5-3  生成器推断方式对照（navtest，one-stage fallback，FM ep18）"))
    E.append(
        note(
            "注：分数已×100 报为百分制。FM first 为三随机种子均值；medoid 基于 100 条样本；"
            "零噪声路径为第 3 章主推交付方式。涨分主要来自 DAC 与两帧扩展舒适 EC。"
        )
    )
    E.append(
        body(
            "表 5-3 显示：官方 DDPM first（K=100）为 73.17；FM first 为 74.18±0.06；medoid 为 75.82；零噪声路径为 76.57。"
            "相对 FM first，medoid / 零噪声分别 +1.64 / +2.39；相对官方 DDPM，零噪声 +3.40。"
            "这说明在同一流匹配模型上，仅改变取点规则即可稳定提升交付轨迹质量，而零噪声路径把“训练期被中心一致性优化的确定解”直接用于推理，收益最大。"
        )
    )
    E.append(
        body(
            "步数与时延方面，FM 在训练推理网格内扫描 K∈{1,2,3,5,8,10,20}，扩散基线扫描 K∈{10,20,50,100}。"
            "主结果口径取 FM 零噪声 K=20（76.57）对照 DDPM K=100（73.17）：更高总分下积分步数约为 1/5。"
            "按约 8 ms/步标定，FM@K=20≈160 ms，DDPM@K=100≈800 ms，FM 进入更高精度—时延分区。对应柱图与 Pareto 见图 5-1、图 5-2。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-1", "图5-1  推断方式对照：官方 DDPM 与 FM first / medoid / zero（navtest）"))
    E.extend(fig_pair(fig_blocks, "图5-2", "图5-2  生成器精度—时延 Pareto 示意（步数×标定单步时延）"))

    E.append(
        body(
            "同预算对照进一步排除“只是训练更久”的解释。表 5-4 在统一 5 epoch 预算下比较官方 DDPM、FM-base、FM+center 以及修复推理网格后的 Phase1-v2 center。"
            "官方 DDPM 终点（ep04，K=100）为 68.22；FM-base 零噪声为 63.99；FM+center 为 71.25（相对官方 +3.03，相对 base +7.26）；"
            "Phase1-v2 在对齐训练推理网格后，center@ep04/K=20 进一步达到 73.52，同点 base 仅 56.15。"
            "因此，在相同短训预算下，结构化流匹配加中心一致性相对官方扩散配方具有可复核增益；中心一致性是短训期拉开差距的关键监督。"
        )
    )
    E.append(T_budget)
    E.append(caption_para("表 5-4  同预算 5 epoch 生成器对照（navtest，one-stage fallback）"))
    E.append(
        note(
            "注：官方 DDPM 三档 checkpoint 为 ep02/03/04 = 67.59 / 67.07 / 68.22（约 11989 场景，0 fail），表中取终点 ep04。"
            "FM 两臂与官方均自同一官方 ckpt 重初始化轨迹头后短训。成熟 FM ep18 零噪声 76.57 不在同预算列，仅作上限参考。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-3", "图5-3  中心一致性短训对照（5 epoch，zero-noise）"))
    E.extend(fig_pair(fig_blocks, "图5-4", "图5-4  Phase1-v2：FM-base 与 FM+center 随 checkpoint 的 navtest 曲线"))
    E.extend(
        fig_pair(
            fig_blocks,
            "图5-5",
            "图5-5  Phase1-v2 训练曲线（epoch 轴对齐）：Official DDPM / FM-base / FM+center",
        )
    )
    # optional supplemental figs if present
    if any(k.startswith("图5-5b") or k == "图5-5b" for k in fig_blocks):
        E.extend(fig_pair(fig_blocks, "图5-5b", "图5-5b  FM 训练过程 navtest 曲线（补充）"))
    if any(k.startswith("图5-5c") or k == "图5-5c" for k in fig_blocks):
        E.extend(
            fig_pair(
                fig_blocks,
                "图5-5c",
                "图5-5c  生成器主结果汇总（官方 DDPM / FM 推断方式 / 中心一致性）",
            )
        )

    E.append(
        body(
            "需要指出，盲目加长训练并不自动更好：center 臂在 18 epoch 长训至 ep17 时总分为 66.21，两帧舒适崩溃至 16.21；"
            "同臂 ep04 早停为 69.58 / 舒适 89.18。因此生成器主推“短训 + 早停 + 零噪声交付”，而非无验证地延长 epoch。"
        )
    )

    # ---- 5.3.4 system ----
    E.append(heading("系统级串联说明"))
    E.append(
        body(
            "将第 3 章生成器与第 4 章评价器串联时，系统配置为：FM-DP 以零噪声路径生成少量连续提案，与 8192 条词表候选合并，"
            "再由 OPD 学生（ResNet34）完成 Top-256 粗筛—精排，并可叠加安全门（ratio=0.8）。"
            "两端的独立评测已经分别证明：生成侧提供更稳的连续提案（表 5-3、表 5-4），评价侧提供安全偏好排序（表 5-1、表 5-5）。"
            "二者互补而非互相替代——生成器负责“提出更好的连续轨迹”，评价器负责“在离散候选中选出更安全的交付”。"
            "定性串联效果见 5.5 节决策可视化。"
        )
    )

    # ============================================================
    # 5.4 消融
    # ============================================================
    E.append(heading("消融实验分析"))
    E.append(
        body(
            "本节在主结果基础上，用已完成的对照实验回答两个问题：生成器涨分来自哪些可分离因素；评价器涨分中蒸馏与安全门各自贡献多少。"
            "消融只报告已经跑完、可写入表格的结果，不把未展开的超参网格写成正文条目。"
        )
    )

    E.append(heading("生成器：推断方式、中心一致性与训练预算"))
    E.append(
        body(
            "生成器消融由三组已完成对照构成，结论可直接与表 5-3、表 5-4 及图 5-1～图 5-5 对应。"
        )
    )
    E.append(
        body(
            "（1）推断方式（同 ep18 checkpoint，零训练成本）。DDPM first 73.17 → FM first 74.18 → medoid 75.82 → 零噪声 76.57。"
            "在不改动网络权重的条件下，确定性交付相对随机 first 提升 2.39 分，相对官方提升 3.40 分，说明第 3 章“少步流匹配 + 中心推断”的推理设计本身有效。"
        )
    )
    E.append(
        body(
            "（2）中心一致性短训（同预算 5 epoch，zero-noise，K=20）。FM-base 63.99，FM+center 71.25（Δ=+7.26）；"
            "Phase1-v2 修复步数网格后 center 达 73.52，同点 base 仅 56.15。对比说明：中心一致性显著改善可交付的确定解；"
            "推理步数必须落在训练网格内，否则出现分布外积分，分数不可比。"
        )
    )
    E.append(
        body(
            "（3）官方同预算基线与长训风险。官方 DDPM 5 epoch 终点 68.22，低于同预算 FM+center 71.25，也低于 Phase1-v2 center 73.52；"
            "18 epoch 长训出现 comfort 崩溃。综合三组结果，生成器定稿策略为：流匹配速度场 + 中心一致性短训 + 早停 + 零噪声 K=20 交付。"
        )
    )

    E.append(heading("评价器：蒸馏构成与安全门"))
    E.append(
        body(
            "评价器消融围绕“谁贡献了 +1.88 分”展开，依据表 5-1 的四级拆解与表 5-5 的安全门开关对照。"
        )
    )
    E.append(
        body(
            "（1）蒸馏与增强的相对贡献。自训 80.69 → +扰动增强 80.88（+0.19）→ +异构蒸馏 82.28（再 +1.40）→ 完整方法 82.57。"
            "扰动增强单独收益有限；异构蒸馏贡献了主要升幅。这与第 4 章判断一致：能力增量来自冻结的强教师，而不是来自更多几何扰动。"
        )
    )
    E.append(
        body(
            "（2）安全门后处理。表 5-5 在 5000 / 9000 / 13000 三档步数上比较 ratio=0 与 ratio=0.8。"
            "以 13000 step 为例，开启安全门后 EPDMS 由 82.37 升至 82.57，zero% 由 7.61% 降至 7.16%，同时 EP 由 82.68 略降至 82.18。"
            "三档步数上均呈现“EPDMS 小幅上升、zero% 稳定下降、EP 小幅下降”的同一模式，说明安全门是可复现的“用舒适换安全”后处理，"
            "且不引入新的可学习参数。默认部署取 ratio=0.8。"
        )
    )
    E.append(T_gate)
    E.append(caption_para("表 5-5  安全门后处理消融（仅改 safety_gate_ratio）"))
    E.append(
        note(
            "注：同一 OPD checkpoint 上仅切换安全门开关；增益主要来自 zero% 下降而非舒适侧提升，"
            "EP 一致小幅下降，符合后处理定位。"
        )
    )
    E.append(
        body(
            "（3）词表规模。全文主实验统一 N=8192，与教师缓存及 Top-K 精排协议一致。"
            "词表 PCA（图 5-9）显示该规模下直行 / 左右转 / 制动意图已形成稳定簇结构；过小易损模态覆盖，过大抬高精排成本，故以 8192 作为定稿折中。"
        )
    )

    # ============================================================
    # 5.5 visualization
    # ============================================================
    E.append(heading("决策过程可视化分析"))
    E.append(
        body(
            "定量指标回答“好多少”，可视化回答“好在何处”。本节从词表语义组织、单场景轨迹对比与多场景对照板三方面给出证据。"
        )
    )
    E.append(heading("词表空间的意图结构"))
    E.append(
        body(
            "对 8192 条人类轨迹聚类词表做主成分分析：前两主成分累计解释约 99.7% 方差（PC1≈95.8%，PC2≈3.9%）。"
            "按直行 / 左拐 / 右拐 / 制动着色后，投影呈现清晰簇结构——直行沿 PC1 铺展，左右转在 PC2 分离，制动集中于一端。"
            "这说明离散决策空间具备可解释的意图覆盖，为粗筛—精排提供了语义上可分的候选集合（图 5-9）。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-9", "图5-9  候选词表 PCA 语义聚类（N=8192）"))

    E.append(heading("生成侧与全链路场景对照"))
    E.append(
        body(
            "在物理空间 BEV / 前视 PV 上叠加 Human、FM 零噪声 / medoid / first、DDPM 与多样本云。"
            "筛选原则为：导航指令一致的前提下，保留 FM 贴近可行驶正确路径且明显优于 DDPM 的样本；若 Human 走错而 FM 走对亦可作为正例。"
            "图 5-10 给出典型左转：FM 贴合 Human 正确路径，DDPM 切弯偏入交叉口中心。图 5-11 的多场景板进一步显示，"
            "DDPM 更易出现切弯、偏航或冲入对向等失败模式，而 FM 提案更稳定地贴近正确路径。"
            "这与第 3 章确定性中心推断、第 4 章安全门精排的设计目标一致：生成侧提供连续且居中的提案，评价侧抑制不安全候选。"
        )
    )
    E.extend(fig_pair(fig_blocks, "图5-10", "图5-10  决策可视化（BEV|PV）：FM 贴合 Human，DDPM 偏差例"))
    E.extend(
        fig_pair(
            fig_blocks,
            "图5-11",
            "图5-11  多场景决策对照板（仅保留 FM 优于或贴合正确路径的样本）",
        )
    )

    # ============================================================
    # 5.6 limitations
    # ============================================================
    E.append(heading("局限性讨论"))
    E.append(
        body(
            "本章结论建立在已完成对照之上，仍有三点局限需要说明。"
            "其一，On-Policy 状态递推的横向残差无法被图像观测如实反映，教师始终看到真实车道位置的图像，故递推时域需受限。"
            "其二，安全门依赖安全头自身的概率标定：它是重排序约束而非新模块，若安全头有偏，门控收益会减弱，因此正文始终与关闭门对照组同报。"
            "其三，主表以单次训练、单次评测的点估计为主，测试集采样不确定性可用 bootstrap 描述，但训练随机性的多种子方差需另行报告。"
        )
    )

    # ============================================================
    # 5.7 summary
    # ============================================================
    E.append(heading("本章小结"))
    E.append(
        body(
            "本章在统一的 NAVSIM / navtest 协议下完成了配置说明、基线对照、主结果、消融与可视化，主要结论如下。"
        )
    )
    E.append(
        body(
            "第一，评价器方面，分解式异构蒸馏是主增益来源：ResNet34 学生由自训 80.69 提升至完整方法 82.57（+1.88），"
            "与 ViT-Large 教师的差距缩小到 4.53；四级拆解表明扰动增强仅 +0.19，蒸馏贡献主要升幅。安全门在三档步数上稳定降低 zero%，"
            "并以可接受的 EP 代价换取更高 EPDMS。学生推理仅 37.9 M / 101.0 ms / 477 MB，相对教师约 1/8 参数量与约 1/5 显存，可直接替换部署。"
        )
    )
    E.append(
        body(
            "第二，生成器方面，确定性零噪声路径在成熟 checkpoint 上达到 76.57，高于官方 DDPM first 的 73.17（+3.40），"
            "并以约 1/5 积分步进入更优精度—时延分区。同预算 5 epoch 下，官方 DDPM 为 68.22，FM+center 为 71.25，"
            "Phase1-v2 center 为 73.52，证明结构化流匹配与中心一致性的增益不是“多训几轮”的假象；长训 comfort 崩溃则支持短训+早停的定稿策略。"
        )
    )
    E.append(
        body(
            "第三，决策可视化表明 8192 词表按驾驶意图清晰分簇，且在筛选场景中 FM 提案更贴近正确路径、DDPM 更易切弯偏航，"
            "与定量结论相互印证。综合以上，第 3、4 章方法在 NAVSIM 上形成了“生成更稳、评价更安全、部署更轻”的完整证据链。"
        )
    )

    return E


def main():
    if not BACKUP.exists():
        shutil.copy2(DOC, BACKUP)
        print("backup ->", BACKUP)
    else:
        # refresh working copy from backup if rewriting again? keep existing backup, copy current to timestamped
        ts_bak = DOC.with_name("硕士论文1112 修订.ch5pre_rewrite.docx")
        shutil.copy2(DOC, ts_bak)
        print("pre-rewrite snapshot ->", ts_bak)

    doc = Document(str(DOC))
    start, end = find_ch5_range(doc)
    print("located Ch5 range")

    table_by_header, fig_blocks = detach_blocks(doc, start, end)
    print("tables:", list(table_by_header))
    print("figs:", sorted(fig_blocks))

    # also remove the old title paragraph (start) — rebuild includes new title
    # start was already removed inside detach_blocks when iterating from start inclusive
    # Wait: detach starts AT start and removes it. Good.
    # But end (参考文献) stays. Insert new content before end.

    elements = build_ch5(table_by_header, fig_blocks)
    append_sequence(end, elements)
    print("inserted", len(elements), "elements")

    doc.save(str(DOC))
    print("saved", DOC)

    # verify
    doc2 = Document(str(DOC))
    start2, end2 = find_ch5_range(doc2)
    texts = []
    grab = False
    for p in doc2.paragraphs:
        t = para_text(p)
        if t == "实验结果与决策过程分析":
            grab = True
        if grab:
            if t == "参考文献":
                break
            if t:
                texts.append(t[:100])
    print("--- Ch5 outline (non-empty paras, truncated) ---")
    for t in texts:
        if len(t) < 40 or t.startswith("表") or t.startswith("图") or t.endswith("分析") or t.endswith("说明") or t.endswith("对照") or t.endswith("配置") or t.endswith("协议") or t.endswith("指标") or t.endswith("方法") or t.endswith("结果") or t.endswith("讨论") or t.endswith("结构"):
            print("*", t)
    print("total non-empty paras in ch5:", len(texts))

    # ensure no wishlist remnants
    bad = ["分别将", "置零，考察三路", "未在本章展开", "不另开全网格", "11. 安全门"]
    blob = "\n".join(texts)
    for b in bad:
        if b in blob:
            print("[warn] leftover:", b)


if __name__ == "__main__":
    main()
