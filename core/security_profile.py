"""Layer 1, deterministic "what is this security" reference data.

No network calls and no LLM calls: resolve_security_profile is a pure
dictionary lookup against a small local snapshot. A ticker absent from the
snapshot resolves to None -- the caller must render a fail-closed "no
profile" message instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

# ticker: (name_zh, name_en, sector_zh_en, industry_zh, description_zh)
_STOCK_PROFILES: dict[str, tuple[str, str, str, str, str]] = {
    "NVDA": ("英伟达", "NVIDIA Corporation", "信息技术（Information Technology）", "半导体",
             "全球主要 GPU 与加速计算平台公司，核心业务覆盖数据中心、AI 计算、游戏和专业图形。"),
    "AAPL": ("苹果", "Apple Inc.", "信息技术（Information Technology）", "科技硬件与消费电子",
             "设计、制造并销售 iPhone、Mac、iPad 等消费电子产品，并提供软件与服务生态。"),
    "MSFT": ("微软", "Microsoft Corporation", "信息技术（Information Technology）", "系统软件",
             "全球主要操作系统、生产力软件与云计算服务提供商，Azure 是核心增长引擎。"),
    "GOOGL": ("谷歌（Alphabet A类）", "Alphabet Inc. Class A", "信息技术（Information Technology）", "互联网服务",
              "全球最大搜索引擎与在线广告平台母公司，业务涵盖云计算、YouTube 与自动驾驶等。"),
    "GOOG": ("谷歌（Alphabet C类）", "Alphabet Inc. Class C", "信息技术（Information Technology）", "互联网服务",
             "全球最大搜索引擎与在线广告平台母公司，业务涵盖云计算、YouTube 与自动驾驶等。"),
    "META": ("Meta Platforms", "Meta Platforms, Inc.", "信息技术（Information Technology）", "互动媒体与社交",
             "旗下拥有 Facebook、Instagram、WhatsApp 等平台，核心业务为社交广告与元宇宙投入。"),
    "AVGO": ("博通", "Broadcom Inc.", "信息技术（Information Technology）", "半导体",
             "全球领先的半导体与基础设施软件公司，产品覆盖网络芯片、存储与定制 AI 芯片。"),
    "AMD": ("超威半导体", "Advanced Micro Devices, Inc.", "信息技术（Information Technology）", "半导体",
            "设计 CPU、GPU 与数据中心加速芯片，是 AI 计算领域的主要参与者之一。"),
    "AMZN": ("亚马逊", "Amazon.com, Inc.", "非必需消费品（Consumer Discretionary）", "互联网零售",
             "全球最大电商平台之一，AWS 云计算业务是公司主要利润来源。"),
    "TSLA": ("特斯拉", "Tesla, Inc.", "非必需消费品（Consumer Discretionary）", "汽车制造",
             "电动汽车与能源存储公司，同时布局自动驾驶与机器人技术。"),
    "NFLX": ("奈飞", "Netflix, Inc.", "通讯服务（Communication Services）", "娱乐",
             "全球领先的流媒体视频订阅服务提供商，业务覆盖原创内容制作与全球发行。"),
    "JPM": ("摩根大通", "JPMorgan Chase & Co.", "金融（Financials）", "多元化银行",
            "美国最大银行控股公司之一，业务涵盖零售银行、投资银行与资产管理。"),
    "V": ("维萨", "Visa Inc.", "金融（Financials）", "支付服务",
          "全球领先的电子支付网络公司，主要通过交易手续费获得收入，不直接放贷。"),
    "BRK-B": ("伯克希尔哈撒韦（B类）", "Berkshire Hathaway Inc. Class B", "金融（Financials）", "多元化控股",
              "由巴菲特领导的多元化控股公司，业务涵盖保险、铁路、能源与大量股票投资。"),
    "UNH": ("联合健康集团", "UnitedHealth Group Incorporated", "医疗保健（Healthcare）", "医疗保健计划",
            "美国最大的健康保险与医疗服务公司之一，业务涵盖保险计划与医疗数据服务。"),
    "LLY": ("礼来", "Eli Lilly and Company", "医疗保健（Healthcare）", "制药",
            "全球领先制药公司，在糖尿病、肥胖症与肿瘤治疗领域具有重要产品线。"),
    "XOM": ("埃克森美孚", "Exxon Mobil Corporation", "能源（Energy）", "石油与天然气",
            "全球最大的综合性石油天然气公司之一，业务覆盖油气勘探、炼化与化工产品。"),
}

# ticker: (official_name_en, asset_class_zh, fund_category_zh, description_zh)
_ETF_PROFILES: dict[str, tuple[str, str, str, str]] = {
    "VOO": ("Vanguard S&P 500 ETF", "美国股票", "标普 500 大型股指数 ETF",
            "跟踪标普 500 指数，覆盖美国大型上市公司，是最常用的美股核心配置工具之一。"),
    "SPY": ("SPDR S&P 500 ETF Trust", "美国股票", "标普 500 大型股指数 ETF",
            "跟踪标普 500 指数，是历史最悠久、交易最活跃的美股 ETF 之一。"),
    "IVV": ("iShares Core S&P 500 ETF", "美国股票", "标普 500 大型股指数 ETF",
            "跟踪标普 500 指数，费率较低，常用于长期核心持仓。"),
    "VTI": ("Vanguard Total Stock Market ETF", "美国股票", "全市场股票指数 ETF",
            "覆盖美国大中小盘几乎全部上市公司，提供最广泛的美股市场敞口。"),
    "QQQ": ("Invesco QQQ Trust", "美国股票", "纳斯达克 100 指数 ETF",
            "跟踪纳斯达克 100 指数，成分股集中于科技与成长型公司。"),
    "QQQM": ("Invesco NASDAQ 100 ETF", "美国股票", "纳斯达克 100 指数 ETF",
             "跟踪纳斯达克 100 指数，费率低于 QQQ，定位为长期持有版本。"),
    "XLK": ("Technology Select Sector SPDR Fund", "美国股票", "信息技术行业 ETF",
            "跟踪标普 500 中信息技术板块成分股，集中于大型科技公司。"),
    "SMH": ("VanEck Semiconductor ETF", "美国股票", "半导体行业 ETF",
            "聚焦全球半导体设计与制造龙头公司，行业集中度较高。"),
    "SOXX": ("iShares Semiconductor ETF", "美国股票", "半导体行业 ETF",
             "跟踪美国半导体行业指数，成分股集中于芯片设计与制造公司。"),
    "VUG": ("Vanguard Growth ETF", "美国股票", "大盘成长风格 ETF",
            "聚焦美国大盘成长股，科技与消费类公司权重较高。"),
    "MTUM": ("iShares MSCI USA Momentum Factor ETF", "美国股票", "动量因子 ETF",
             "持仓基于价格动量因子筛选，成分股会随动量变化定期调整。"),
    "SPMO": ("Invesco S&P 500 Momentum ETF", "美国股票", "大型股动量因子 ETF",
             "从标普 500 成分股中筛选动量最强的部分，行业集中度会随市场轮动变化。"),
    "VTV": ("Vanguard Value ETF", "美国股票", "大盘价值风格 ETF",
            "聚焦美国大盘价值股，金融与医疗保健等板块权重相对较高。"),
    "SCHD": ("Schwab U.S. Dividend Equity ETF", "美国股票", "高股息质量因子 ETF",
             "聚焦具有持续分红记录与财务质量的美股公司。"),
    "VYM": ("Vanguard High Dividend Yield ETF", "美国股票", "高股息 ETF",
            "聚焦高股息收益率的美国上市公司，覆盖多个行业。"),
    "VHT": ("Vanguard Health Care ETF", "美国股票", "医疗保健行业 ETF",
            "跟踪美国医疗保健板块，涵盖制药、医疗器械与医疗服务公司。"),
    "XLV": ("Health Care Select Sector SPDR Fund", "美国股票", "医疗保健行业 ETF",
            "跟踪标普 500 中医疗保健板块成分股。"),
    "XLF": ("Financial Select Sector SPDR Fund", "美国股票", "金融行业 ETF",
            "跟踪标普 500 中金融板块成分股，涵盖银行、保险与资本市场公司。"),
    "XLE": ("Energy Select Sector SPDR Fund", "美国股票", "能源行业 ETF",
            "跟踪标普 500 中能源板块成分股，主要为石油天然气公司。"),
    "XLI": ("Industrial Select Sector SPDR Fund", "美国股票", "工业行业 ETF",
            "跟踪标普 500 中工业板块成分股，涵盖航空、机械与运输公司。"),
    "XLU": ("Utilities Select Sector SPDR Fund", "美国股票", "公用事业行业 ETF",
            "跟踪标普 500 中公用事业板块成分股，波动性相对较低。"),
    "XLP": ("Consumer Staples Select Sector SPDR Fund", "美国股票", "必需消费品行业 ETF",
            "跟踪标普 500 中必需消费品板块成分股。"),
    "XLY": ("Consumer Discretionary Select Sector SPDR Fund", "美国股票", "非必需消费品行业 ETF",
            "跟踪标普 500 中非必需消费品板块成分股。"),
    "GRID": ("First Trust NASDAQ Clean Edge Smart Grid Infrastructure ETF", "美国股票", "智能电网与基础设施主题 ETF",
             "聚焦智能电网、储能与电力基础设施相关公司。"),
    "VXUS": ("Vanguard Total International Stock ETF", "国际股票", "全球（除美国）股票指数 ETF",
             "覆盖美国以外的发达与新兴市场股票，提供国际市场分散敞口。"),
    "VEA": ("Vanguard FTSE Developed Markets ETF", "国际股票", "发达市场（除美国）股票指数 ETF",
            "覆盖欧洲、日本等发达市场上市公司。"),
    "VWO": ("Vanguard FTSE Emerging Markets ETF", "国际股票", "新兴市场股票指数 ETF",
            "覆盖中国、印度、巴西等新兴市场上市公司。"),
    "EFA": ("iShares MSCI EAFE ETF", "国际股票", "发达市场（除美加）股票指数 ETF",
            "覆盖欧洲、澳洲及远东发达市场上市公司。"),
    "ZSP": ("BMO S&P 500 Index ETF", "美国股票（加元计价）", "标普 500 大型股指数 ETF",
            "以加元计价跟踪标普 500 指数，为加拿大投资者提供美股核心敞口。"),
    "VDY": ("Vanguard FTSE Canadian High Dividend Yield Index ETF", "加拿大股票", "高股息 ETF",
            "聚焦加拿大高股息收益率上市公司，金融与能源板块权重较高。"),
    "SGOV": ("iShares 0-3 Month Treasury Bond ETF", "固定收益", "美国超短期国债",
             "主要投资剩余期限不超过约 3 个月的美国国债，常用于短期资金管理、降低组合波动和获取短期国债收益。"),
    "BIL": ("SPDR Bloomberg 1-3 Month T-Bill ETF", "固定收益", "美国超短期国债",
            "投资于 1-3 个月期限的美国短期国债，波动性极低，常作现金替代品使用。"),
    "SHY": ("iShares 1-3 Year Treasury Bond ETF", "固定收益", "美国短期国债",
            "投资于剩余期限 1-3 年的美国国债，久期较短，利率敏感度低于长期债券。"),
    "AGG": ("iShares Core U.S. Aggregate Bond ETF", "固定收益", "美国综合债券指数 ETF",
            "覆盖美国投资级国债、机构债与公司债，是常用的核心债券配置工具。"),
    "BND": ("Vanguard Total Bond Market ETF", "固定收益", "美国综合债券指数 ETF",
            "覆盖美国投资级应税债券市场，久期与信用分布较为分散。"),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "固定收益", "美国长期国债",
            "投资于剩余期限 20 年以上的美国国债，对利率变化非常敏感。"),
    "CBIL": ("Global X 0-3 Month T-Bill ETF CAD", "固定收益", "加拿大超短期国债",
             "主要投资剩余期限不超过约 3 个月的加拿大国库券，常用于短期资金管理与降低组合波动。"),
    "FINN": ("Fidelity Global Innovators ETF Series L", "全球股票", "全球创新主题股票 ETF",
             "主动管理的全球股票基金，聚焦具备长期增长潜力的创新型公司，覆盖科技、医疗等多个行业。"),
    "XSB": ("iShares Core Canadian Short Term Bond Index ETF", "固定收益", "加拿大短期债券 ETF",
            "投资于加拿大短期政府与公司债券，久期较短，波动性相对较低。"),
    "GLD": ("SPDR Gold Shares", "另类资产", "黄金 ETF",
            "通过持有实物黄金追踪金价表现，常用于组合对冲与多元化配置。"),
    "USO": ("United States Oil Fund", "另类资产", "原油期货 ETF",
            "通过原油期货合约追踪 WTI 原油价格表现，波动性较高。"),
}


@dataclass(frozen=True)
class SecurityProfile:
    ticker: str
    title: str
    kind_label: str
    category_label: str
    category_value: str
    subcategory_label: str
    subcategory_value: str
    description: str


def resolve_security_profile(ticker: str) -> SecurityProfile | None:
    """Deterministic, local-only lookup. Returns None (fail closed) for any
    ticker outside the snapshot -- callers must show "no profile", never a
    guess, and must never call an LLM to fill the gap."""
    key = (ticker or "").strip().upper()
    if key in _STOCK_PROFILES:
        name_zh, name_en, sector, industry, description = _STOCK_PROFILES[key]
        return SecurityProfile(
            key, f"{name_zh}（{name_en}）", "股票（Stock）",
            "板块", sector, "行业", industry, description,
        )
    if key in _ETF_PROFILES:
        name_en, asset_class, fund_category, description = _ETF_PROFILES[key]
        return SecurityProfile(
            key, name_en, "ETF",
            "资产类别", asset_class, "基金类别", fund_category, description,
        )
    return None


def stock_name_en(ticker: str) -> str | None:
    """The canonical English company name for a known stock ticker, from the
    same _STOCK_PROFILES snapshot resolve_security_profile already uses --
    never a guessed/inferred name, never a new data source or network
    lookup. Returns None (fail closed) for any ticker outside the snapshot.
    Used by core.etf_holdings (Step 2A.11) to label ETF constituent rows."""
    entry = _STOCK_PROFILES.get((ticker or "").strip().upper())
    return entry[1] if entry else None
