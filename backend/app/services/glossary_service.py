"""专有名词（词典）服务"""
from __future__ import annotations

import json
from typing import Optional

from app.models.schemas import GlossaryItem
from app.storage.database import Database

# 内置词库：常见动漫专有名词、设定词和漫画口语（日语→中文）
BUILTIN_GLOSSARY = [
    # 人物（character）：保留日文名、常见罗马字/别名的独立映射，适配当前 source→target 结构。
    {"source": "ナルト", "target": "鸣人", "lang": "ja", "note": "[character]《火影忍者》角色"},
    {"source": "うずまきナルト", "target": "漩涡鸣人", "lang": "ja", "note": "[character] 全名"},
    {"source": "サスケ", "target": "佐助", "lang": "ja", "note": "[character]《火影忍者》角色"},
    {"source": "うちはサスケ", "target": "宇智波佐助", "lang": "ja", "note": "[character] 全名"},
    {"source": "ルフィ", "target": "路飞", "lang": "ja", "note": "[character]《海贼王》角色"},
    {"source": "ゴッドバレー", "target": "神之谷", "lang": "ja", "note": "[location]《海贼王》地点"},
    {"source": "ゴッド・バレー", "target": "神之谷", "lang": "ja", "note": "[location]《海贼王》地点写法"},
    {"source": "小僧共", "target": "小鬼们", "lang": "ja", "note": "[colloquial] 漫画口语，整体翻译"},
    {"source": "うちに寄ってきなよ", "target": "来我家坐坐吧", "lang": "ja", "note": "[colloquial] 邀请语"},
    {"source": "鼻緒", "target": "木屐带", "lang": "ja", "note": "[item] 木屐的带子"},
    {"source": "モンキー・D・ルフィ", "target": "蒙奇·D·路飞", "lang": "ja", "note": "[character] 全名"},
    {"source": "ゾロ", "target": "索隆", "lang": "ja", "note": "[character]《海贼王》角色"},
    {"source": "ナミ", "target": "娜美", "lang": "ja", "note": "[character]《海贼王》角色"},
    {"source": "サンジ", "target": "山治", "lang": "ja", "note": "[character]《海贼王》角色"},
    {"source": "トニートニー・チョッパー", "target": "托尼托尼·乔巴", "lang": "ja", "note": "[character] 全名"},
    {"source": "エドワード・ニューゲート", "target": "爱德华·纽盖特", "lang": "ja", "note": "[character] 白胡子全名"},
    {"source": "孫悟空", "target": "孙悟空", "lang": "ja", "note": "[character]《龙珠》角色"},
    {"source": "ベジータ", "target": "贝吉塔", "lang": "ja", "note": "[character]《龙珠》角色"},
    {"source": "ブルマ", "target": "布尔玛", "lang": "ja", "note": "[character]《龙珠》角色"},
    {"source": "竈門炭治郎", "target": "灶门炭治郎", "lang": "ja", "note": "[character]《鬼灭之刃》角色"},
    {"source": "竈門禰豆子", "target": "灶门祢豆子", "lang": "ja", "note": "[character]《鬼灭之刃》角色"},
    {"source": "我妻善逸", "target": "我妻善逸", "lang": "ja", "note": "[character]《鬼灭之刃》角色"},
    {"source": "嘴平伊之助", "target": "嘴平伊之助", "lang": "ja", "note": "[character]《鬼灭之刃》角色"},
    {"source": "虎杖悠仁", "target": "虎杖悠仁", "lang": "ja", "note": "[character]《咒术回战》角色"},
    {"source": "伏黒恵", "target": "伏黑惠", "lang": "ja", "note": "[character]《咒术回战》角色"},
    {"source": "五条悟", "target": "五条悟", "lang": "ja", "note": "[character]《咒术回战》角色"},
    {"source": "坂田銀時", "target": "坂田银时", "lang": "ja", "note": "[character]《银魂》角色"},
    {"source": "神楽", "target": "神乐", "lang": "ja", "note": "[character]《银魂》角色"},
    {"source": "日向翔陽", "target": "日向翔阳", "lang": "ja", "note": "[character]《排球少年！！》角色"},
    {"source": "影山飛雄", "target": "影山飞雄", "lang": "ja", "note": "[character]《排球少年！！》角色"},
    {"source": "綾波レイ", "target": "绫波丽", "lang": "ja", "note": "[character]《新世纪福音战士》角色"},
    {"source": "碇シンジ", "target": "碇真嗣", "lang": "ja", "note": "[character]《新世纪福音战士》角色"},
    {"source": "コテ川", "target": "古手川", "lang": "ja", "note": "[character]《To LOVEる》角色，OCR 常见简称"},
    {"source": "天野陽菜", "target": "天野阳菜", "lang": "ja", "note": "[character]《天气之子》角色"},
    {"source": "桜木花道", "target": "樱木花道", "lang": "ja", "note": "[character]《灌篮高手》角色"},
    {"source": "流川楓", "target": "流川枫", "lang": "ja", "note": "[character]《灌篮高手》角色"},
    {"source": "木之本桜", "target": "木之本樱", "lang": "ja", "note": "[character]《魔卡少女樱》角色"},
    {"source": "月野うさぎ", "target": "月野兔", "lang": "ja", "note": "[character]《美少女战士》角色"},
    # 地点（location）
    {"source": "木ノ葉隠れの里", "target": "木叶隐村", "lang": "ja", "note": "[location]《火影忍者》村落"},
    {"source": "砂隠れの里", "target": "砂隐村", "lang": "ja", "note": "[location]《火影忍者》村落"},
    {"source": "海軍本部", "target": "海军本部", "lang": "ja", "note": "[location]《海贼王》地点"},
    {"source": "偉大なる航路", "target": "伟大航路", "lang": "ja", "note": "[location]《海贼王》海域"},
    {"source": "新世界", "target": "新世界", "lang": "ja", "note": "[location]《海贼王》海域"},
    {"source": "空島", "target": "空岛", "lang": "ja", "note": "[location]《海贼王》地点"},
    {"source": "米花町", "target": "米花町", "lang": "ja", "note": "[location]《名侦探柯南》虚构街区"},
    {"source": "東京都", "target": "东京都", "lang": "ja", "note": "[location] 城市/行政区"},
    {"source": "渋谷", "target": "涩谷", "lang": "ja", "note": "[location] 东京地区"},
    {"source": "新宿", "target": "新宿", "lang": "ja", "note": "[location] 东京地区"},
    {"source": "烏野高校", "target": "乌野高中", "lang": "ja", "note": "[location]《排球少年！！》学校"},
    {"source": "箱根", "target": "箱根", "lang": "ja", "note": "[location]《新世纪福音战士》常见地点"},
    {"source": "鎌倉", "target": "镰仓", "lang": "ja", "note": "[location] 日本地名"},
    {"source": "御茶ノ水", "target": "御茶之水", "lang": "ja", "note": "[location] 东京地区"},
    {"source": "デュエルアカデミア", "target": "决斗学院", "lang": "ja", "note": "[location]《游戏王GX》学校"},
    {"source": "ホグワーツ", "target": "霍格沃茨", "lang": "ja", "note": "[location]《哈利·波特》学校"},
    {"source": "アスガルド", "target": "阿斯加德", "lang": "ja", "note": "[location] 奇幻世界地点"},
    # 组织、阵营、身份（organization/title）
    {"source": "麦わらの一味", "target": "草帽一伙", "lang": "ja", "note": "[organization]《海贼王》海贼团"},
    {"source": "赤髪海賊団", "target": "红发海贼团", "lang": "ja", "note": "[organization]《海贼王》海贼团"},
    {"source": "海軍", "target": "海军", "lang": "ja", "note": "[organization]《海贼王》阵营"},
    {"source": "暁", "target": "晓", "lang": "ja", "note": "[organization]《火影忍者》组织"},
    {"source": "木ノ葉の里", "target": "木叶村", "lang": "ja", "note": "[organization] 忍村简称"},
    {"source": "鬼殺隊", "target": "鬼杀队", "lang": "ja", "note": "[organization]《鬼灭之刃》组织"},
    {"source": "呪術高専", "target": "咒术高专", "lang": "ja", "note": "[organization]《咒术回战》学校组织"},
    {"source": "死神代行", "target": "代理死神", "lang": "ja", "note": "[title]《死神》身份"},
    {"source": "魔法騎士団", "target": "魔法骑士团", "lang": "ja", "note": "[organization]《黑色五叶草》组织类型"},
    {"source": "調査兵団", "target": "调查兵团", "lang": "ja", "note": "[organization]《进击的巨人》组织"},
    {"source": "幻影旅団", "target": "幻影旅团", "lang": "ja", "note": "[organization]《全职猎人》组织"},
    {"source": "十二鬼月", "target": "十二鬼月", "lang": "ja", "note": "[organization]《鬼灭之刃》阵营"},
    # 能力、设定、物品（ability/item）
    {"source": "螺旋丸", "target": "螺旋丸", "lang": "ja", "note": "[ability]《火影忍者》忍术"},
    {"source": "千鳥", "target": "千鸟", "lang": "ja", "note": "[ability]《火影忍者》忍术"},
    {"source": "影分身の術", "target": "影分身之术", "lang": "ja", "note": "[ability]《火影忍者》忍术"},
    {"source": "ゴムゴムの実", "target": "橡胶果实", "lang": "ja", "note": "[ability]《海贼王》恶魔果实"},
    {"source": "悪魔の実", "target": "恶魔果实", "lang": "ja", "note": "[ability]《海贼王》设定"},
    {"source": "覇気", "target": "霸气", "lang": "ja", "note": "[ability]《海贼王》战斗体系"},
    {"source": "超サイヤ人", "target": "超级赛亚人", "lang": "ja", "note": "[ability]《龙珠》形态"},
    {"source": "かめはめ波", "target": "龟派气功", "lang": "ja", "note": "[ability]《龙珠》招式"},
    {"source": "領域展開", "target": "领域展开", "lang": "ja", "note": "[ability]《咒术回战》术式"},
    {"source": "全集中の呼吸", "target": "全集中之呼吸", "lang": "ja", "note": "[ability]《鬼灭之刃》战斗体系"},
    {"source": "水の呼吸", "target": "水之呼吸", "lang": "ja", "note": "[ability]《鬼灭之刃》流派"},
    {"source": "卍解", "target": "卍解", "lang": "ja", "note": "[ability]《死神》能力形态"},
    {"source": "念能力", "target": "念能力", "lang": "ja", "note": "[ability]《全职猎人》能力体系"},
    {"source": "スタンド", "target": "替身", "lang": "ja", "note": "[ability]《JOJO的奇妙冒险》能力体系"},
    {"source": "波紋", "target": "波纹", "lang": "ja", "note": "[ability]《JOJO的奇妙冒险》能力体系"},
    {"source": "魔法石", "target": "魔法石", "lang": "ja", "note": "[item] 奇幻设定物品，具体含义需结合作品"},
    {"source": "斬魄刀", "target": "斩魄刀", "lang": "ja", "note": "[item]《死神》武器"},
    {"source": "草帽", "target": "草帽", "lang": "ja", "note": "[item]《海贼王》标志性物品"},
    {"source": "写輪眼", "target": "写轮眼", "lang": "ja", "note": "[ability]《火影忍者》血继限界"},
    {"source": "巨人化", "target": "巨人化", "lang": "ja", "note": "[ability]《进击的巨人》设定"},
    {"source": "冒険者ギルド", "target": "冒险者公会", "lang": "ja", "note": "[organization] 奇幻作品常见组织"},
    # 日语漫画口语/俗语（idiom/colloquial）：仅对稳定短表达固定翻译，歧义项标注结合上下文。
    {"source": "やれやれ", "target": "真是的", "lang": "ja", "note": "[idiom] 也可译为‘哎呀呀’，结合语气"},
    {"source": "しょうがない", "target": "没办法", "lang": "ja", "note": "[idiom] 常见口语"},
    {"source": "ふざけるな", "target": "别开玩笑了", "lang": "ja", "note": "[colloquial] 强烈斥责，视语境可译‘少胡闹’"},
    {"source": "まさか", "target": "不会吧", "lang": "ja", "note": "[colloquial] 惊讶/否定推测，需结合上下文"},
    {"source": "なるほど", "target": "原来如此", "lang": "ja", "note": "[colloquial] 也可译‘明白了’"},
    {"source": "さすが", "target": "不愧是", "lang": "ja", "note": "[colloquial] 后接对象时使用；单独出现需结合语境"},
    {"source": "任せろ", "target": "交给我吧", "lang": "ja", "note": "[colloquial] 也可译‘包在我身上’"},
    {"source": "気をつけろ", "target": "小心", "lang": "ja", "note": "[colloquial] 警告语"},
    {"source": "ありえない", "target": "不可能", "lang": "ja", "note": "[colloquial] 惊讶时也可译‘难以置信’"},
    {"source": "ちくしょう", "target": "可恶", "lang": "ja", "note": "[idiom] 懊恼/骂语，程度需结合语境"},
    {"source": "どういうこと", "target": "怎么回事", "lang": "ja", "note": "[colloquial] 疑问句短语"},
    {"source": "本当か", "target": "真的吗", "lang": "ja", "note": "[colloquial] 也可译‘当真吗’，结合角色语气"},
    {"source": "信じられない", "target": "难以置信", "lang": "ja", "note": "[colloquial] 也可译‘不敢相信’"},
    {"source": "大丈夫", "target": "没事吧", "lang": "ja", "note": "[colloquial] 问句常用译法；肯定回答时可译‘没问题’"},
    {"source": "落ち着け", "target": "冷静点", "lang": "ja", "note": "[colloquial] 命令/安抚语气"},
    {"source": "急げ", "target": "快点", "lang": "ja", "note": "[colloquial] 命令语气"},
    {"source": "逃げろ", "target": "快逃", "lang": "ja", "note": "[colloquial] 紧急警告"},
    {"source": "黙れ", "target": "闭嘴", "lang": "ja", "note": "[colloquial] 粗鲁命令，需结合人物语气"},
    {"source": "なぜだ", "target": "为什么", "lang": "ja", "note": "[colloquial] 也可译‘为何’"},
    {"source": "頼む", "target": "拜托了", "lang": "ja", "note": "[colloquial] 请求/托付，结合上下文"},
    {"source": "勝手にしろ", "target": "随你的便", "lang": "ja", "note": "[idiom] 赌气语气，非字面‘随便做’"},
    {"source": "冗談じゃない", "target": "开什么玩笑", "lang": "ja", "note": "[idiom] 也可译‘才不是玩笑’"},
    {"source": "覚えてろ", "target": "你给我记住", "lang": "ja", "note": "[idiom] 反派退场常见威胁语，需结合上下文"},
    {"source": "お疲れ様", "target": "辛苦了", "lang": "ja", "note": "[idiom] 致意语，正式度可按场景调整"},
]

# 英文漫画表达（English→简体中文）；与日文内置词库一样通过普通 source/target 映射生效。
BUILTIN_ENGLISH_GLOSSARY = {
    "Ken Takakura": "高仓健",
    "Monkey D. Luffy": "蒙奇·D·路飞",
    "Naruto Uzumaki": "漩涡鸣人",
    "Sasuke Uchiha": "宇智波佐助",
    "No way": "不会吧",
    "Damn it": "可恶",
    "Leave it to me": "交给我吧",
    "Watch out": "小心",
    "Give me a break": "饶了我吧",
    "You wish": "你想得美",
    "Seriously?": "认真的吗？",
    "What the hell": "搞什么鬼",
    "Come on": "拜托／得了吧",
    "Not a chance": "没门",
    "I owe you one": "我欠你一个人情",
    "Bring it on": "放马过来",
    "Hang in there": "坚持住",
}


class GlossaryService:
    """专有名词管理，含内置词库 + 用户自定义"""

    def __init__(self):
        self.db = Database.get_instance()

    def _ensure_data(self) -> None:
        """将系统词库作为普通用户词条一次性初始化，并迁移历史内置标记。

        本系统不再维护"系统内置不可删"词库；BUILTIN_GLOSSARY 与 BUILTIN_ENGLISH_GLOSSARY
        仅作为首次部署时的默认用户数据写入，之后所有词条均可由用户增删改查。
        """
        if self.db.glossary_get_meta("initial_data_seeded") == "1":
            return
        # 对已有数据的数据库，视为已完成初始化，避免覆盖用户删除操作。
        total = self.db._connect().execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
        if total > 0:
            self.db.glossary_set_meta("initial_data_seeded", "1")
            return
        for item in BUILTIN_GLOSSARY:
            self.db.glossary_create(
                item["source"], item["target"], item["lang"], item.get("note", ""), "zh", builtin=False
            )
        for source, target in BUILTIN_ENGLISH_GLOSSARY.items():
            self.db.glossary_create(
                source, target, "en", "[builtin] 英文漫画常见表达", "zh", builtin=False
            )
        # 历史版本中 builtin=1 的词条全部转为普通用户词条，使删除功能对它们生效。
        self.db.glossary_migrate_builtin_to_user()
        self.db.glossary_set_meta("initial_data_seeded", "1")

    def list_items(self, lang: Optional[str] = None, search: str = "") -> list[dict]:
        self._ensure_data()
        return self.db.glossary_list(lang=lang, search=search)

    def create(
        self, source: str, target: str, lang: str, note: str = "", target_lang: str = "zh"
    ) -> tuple[int, str]:
        """返回 (id, 消息)。id=-1 表示重复"""
        self._ensure_data()
        item_id = self.db.glossary_create(source, target, lang, note, target_lang)
        if item_id == -1:
            return -1, "该词条已存在"
        return item_id, "已添加"

    def update(
        self, item_id: int, source: str, target: str, lang: str, note: str = "", target_lang: str = "zh"
    ) -> tuple[bool, str]:
        ok = self.db.glossary_update(item_id, source, target, lang, note, target_lang)
        return ok, "已更新" if ok else "更新失败或词条冲突"

    def delete(self, item_id: int) -> bool:
        return self.db.glossary_delete(item_id)

    def get_mapping(self, lang: str, target_lang: str = "zh") -> dict[str, str]:
        """返回 {source: target} 映射（用户词库）"""
        self._ensure_data()
        return self.db.glossary_get_mapping(lang, target_lang)

    def import_json(self, content: str) -> dict:
        """从 JSON 导入词条：[{source,target,lang,note}, ...]"""
        errors: list[str] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"imported": 0, "skipped": 0, "errors": [f"JSON 格式错误: {e}"]}

        if not isinstance(data, list):
            return {"imported": 0, "skipped": 0, "errors": ["JSON 必须为数组"]}

        valid_items = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"第{i+1}项不是对象")
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            if not src or not tgt:
                errors.append(f"第{i+1}项缺少 source 或 target")
                continue
            valid_items.append(
                {
                    "source": src,
                    "target": tgt,
                    "lang": str(item.get("lang", "ja")),
                    "target_lang": str(item.get("target_lang", "zh")),
                    "note": str(item.get("note", "")),
                }
            )

        imported, skipped = self.db.glossary_import(valid_items)
        return {"imported": imported, "skipped": skipped, "errors": errors}
