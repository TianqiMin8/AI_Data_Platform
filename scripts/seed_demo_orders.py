"""
创建并填充测试用业务表 demo_orders（Text-to-SQL 的查询对象）。

demo_orders 不属于 ORM 模型，alembic/env.py 已配置在 autogenerate 时忽略它。
脚本可重复执行：表已存在且有数据时不会重复插入。

用法：
    poetry run python scripts/seed_demo_orders.py
"""

from sqlalchemy import create_engine, text

from app.config.settings import get_settings

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS demo_orders (
    id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    amount NUMERIC(10, 2) NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# 10 笔订单 / 4 个客户 / 3 个品类，总额 22880.50，2 月 12000.50，3 月 4 笔，金额 > 2000 的 5 笔
ORDERS = [
    ("张三", 2100.00, "电子产品", "2026-01-05 10:00:00"),
    ("李四", 1830.00, "服装", "2026-01-12 14:30:00"),
    ("王五", 800.00, "食品", "2026-01-20 09:15:00"),
    ("张三", 5000.00, "电子产品", "2026-02-03 11:00:00"),
    ("赵六", 4500.50, "电子产品", "2026-02-14 16:45:00"),
    ("李四", 2500.00, "服装", "2026-02-22 13:20:00"),
    ("王五", 3200.00, "电子产品", "2026-03-02 10:30:00"),
    ("赵六", 1500.00, "服装", "2026-03-10 15:00:00"),
    ("张三", 800.00, "食品", "2026-03-18 12:00:00"),
    ("李四", 650.00, "食品", "2026-03-25 17:10:00"),
]


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url.replace("+asyncpg", ""))

    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

        existing = conn.execute(text("SELECT COUNT(*) FROM demo_orders")).scalar()
        if existing:
            print(f"demo_orders 已有 {existing} 条数据，跳过插入")
            return

        conn.execute(
            text(
                "INSERT INTO demo_orders (customer_name, amount, category, created_at) "
                "VALUES (:customer_name, :amount, :category, :created_at)"
            ),
            [
                {"customer_name": c, "amount": a, "category": cat, "created_at": ts}
                for c, a, cat, ts in ORDERS
            ],
        )
        print(f"已插入 {len(ORDERS)} 条订单")


if __name__ == "__main__":
    main()
