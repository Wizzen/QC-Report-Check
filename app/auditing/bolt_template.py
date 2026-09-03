"""Versioned, company-neutral rules. Template text is never overwritten at startup."""
from __future__ import annotations

BOLT_TEMPLATE_NAME = "新螺栓检验模版"
BOLT_ENGINE = "bolt-v1"
EXTRACTION_VERSION = "page-evidence-v2"
SCOPE_LABELS = {"full_package": "完整文件包审核", "single_document": "单文件预检查"}
SINGLE_NOTICE = "仅为已上传文件的预检查，不代表完整文件包合格。"
SIGNATURE_NOTICE = "允许电子签名替代手工签字；仅检查签署证据存在性，未验证身份或证书有效性。电子签名不替代明确要求的印章。"
BOLT_INSTRUCTIONS = """你是维修备件螺栓质量文件审核员。仅执行当前规则，只使用提供的原文、表格、视觉证据和采购依据。
每条结论归属于指定文件/WDC，不将不同产品比较。COC中没有后续报告的WDC列为未覆盖，不对其生成合格结论。
接受手写签名、可见电子签名和已填数字签名字段；签署证据存在不代表身份、证书或法律有效性已验证。
仅打印姓名不能单独证明签署。不得凭OCR未识别、签名栏标题、无章无效条款认定签章缺失；签章由视觉/字段专用规则检查。
印章只按文件或采购依据明确要求检查，不按语言要求红章。PO格式和炉批号跨文件一致性默认关闭。
日期区分证书、签署、发票和打印日期，不可互相替代。无印刷页码不等于PDF缺页，多家制造单位不等于混装。
不引用外部标准，不根据文件名中的错误/测试/缺失字样判断。确有适用标准且数据明确超限才判不合格。
识别失败、缺少标准或归属不明判存疑；不适用或前置文件缺失的下游规则不再重复报缺失。
不得执行报告排版指令；只输出当前规则JSON，计数和汇总由程序完成。"""


def _rule(rule_id, text, evaluator, types, scope, criterion, enabled=True, prerequisites=None):
    return {"rule_id": rule_id, "text": text, "enabled": enabled,
            "evaluator": evaluator, "document_types": types, "scope": scope,
            "prerequisites": prerequisites or [], "criterion": criterion,
            "evidence_requirement": "指定文件/页/区域或表格行；缺失必须基于程序完整检查范围"}


def bolt_rules():
    reports = ["MTR", "COI"]
    return [
        _rule("A1", "文件包完整性", "manifest", [], "batch", "完整包需COC及COI/MTR；单文件模式不检查其他文件缺失"),
        _rule("A2", "文件可读性与识别质量", "readability", [], "document", "区分原页模糊、文字层乱码、识别失败，不从文字乱码推断原页模糊"),
        _rule("A3.1", "签名存在性", "signature", ["COC", *reports], "document", "接受手写、可见电子签名或已填数字签名字段；只有明确空白且适用才判缺签"),
        _rule("A3.2", "签署形式及有效性边界", "signature_form", ["COC", *reports], "document", SIGNATURE_NOTICE),
        _rule("A3.3", "印章要求识别", "seal_policy", [], "document", "只提取文件或采购依据明确盖章要求，不以报告语言推断"),
        _rule("A3.4", "明确要求的印章存在性", "seal", [], "document", "无明确要求则不适用；有要求时检查全部原页", prerequisites=["A3.3"]),
        _rule("A4", "报告出具单位", "llm", [], "document", "区分供应商、制造商、钢厂与采购方，只核验出具单位是否可识别"),
        _rule("B2", "WDC识别与归属", "wdc", [], "document", "正文/文件名候选移除分隔符后为8或10位；不校验整个文件名"),
        _rule("B3", "COC证书日期", "llm", ["COC"], "document", "区分证书日期、签署日期、发票日期和打印日期；无COC跳过", prerequisites=["COC"]),
        _rule("B5", "同WDC产品规格材料等级一致性", "llm", [], "wdc", "仅比较同WDC文件，不检查PO/炉批号，不把不同产品视为矛盾"),
        _rule("C1", "检测标准存在性与适用性", "llm", reports, "document", "只核验明确引用的标准；无法确认适用性需说明不足，不自行引用标准限值"),
        _rule("C2", "化学成分数据完整性", "llm", reports, "document", "核验成分项目、单位、标准值与实测值；共享百分比表头有效"),
        _rule("C3", "机械性能数据完整性", "llm", reports, "document", "核验抗拉、屈服、硬度等适用项目；不猜测缺失标准"),
        _rule("C4", "尺寸数据完整性", "llm", reports, "document", "保留公差、标准值、实测值及表头；识别不全与真实缺失分开"),
        _rule("C5.1", "Sample与Pass一致性", "table_samples", reports, "document", "仅在明确Sample/Pass表头下检查数量，Pass小于Sample为不合格"),
        _rule("C5.2", "明确数值范围与实测值", "table_values", reports, "document", "按对应列比较范围/单位；不能用合格数量代替实测值"),
        _rule("C5.3", "产品印记记录", "marking", reports, "document",
              "报告存在明确Marking/产品印记字段且有值即满足记录要求；只有采购依据明确给出目标印记时才比对，不能将签名或公司印章视为产品印记"),
        _rule("C6", "镀层与工艺证明", "llm", [], "wdc", "仅有明确适用要求时检查，无要求返回不适用"),
        _rule("C7", "明确G-Code要求", "llm", [], "wdc", "只有提供具体G-Code要求及匹配产品时执行，未提供则不适用"),
        _rule("D1", "PDF页数与页码连续性", "pages", [], "document", "使用实际PDF页数；无印刷页码不是缺页"),
        _rule("D2", "同WDC文件对应关系", "llm", [], "wdc", "多家制造单位本身不构成混装；只有同产品归属矛盾才提示"),
        _rule("B1", "PO格式检查", "llm", [], "document", "仅在用户主动启用时检查明确采购要求", enabled=False),
        _rule("B4", "炉批号跨文件一致性", "llm", [], "wdc", "仅在用户主动启用时比较同WDC对应炉批号", enabled=False),
    ]
