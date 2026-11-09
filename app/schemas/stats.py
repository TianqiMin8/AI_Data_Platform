from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    hit_count: int
    miss_count: int
    total: int
    hit_rate: float              # 0 ~ 1
    exact_hit_count: int         # 命中中，精确匹配贡献的次数
    semantic_hit_count: int      # 命中中，语义匹配贡献的次数
    warn_threshold: float
    below_threshold: bool        # 样本足够且命中率低于阈值
