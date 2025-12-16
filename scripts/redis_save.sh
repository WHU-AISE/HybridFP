#!/bin/bash

REDIS_HOST="172.17.0.1"
REDIS_PORT="6379"
REDIS_PASSWORD="openwhisk"
OUTPUT_DIR="./logs/redis_export"
RDB_FILE="$OUTPUT_DIR/dump.rdb"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "导出 RDB 文件..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning --rdb "$RDB_FILE"

# 各种导出选项
echo "选择导出格式:"
echo "1. 完整 JSON 导出"
echo "2. 仅键和内存使用情况 (CSV)"
echo "3. 按数据类型过滤导出"
read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        # 完整 JSON 导出
        rdb --command json "$RDB_FILE" > "$OUTPUT_DIR/redis_full.json"
        echo "完整 JSON 导出完成: $OUTPUT_DIR/redis_full.json"
        ;;
    2)
        # 内存分析 CSV
        rdb -c memory "$RDB_FILE" --bytes 128 -f "$OUTPUT_DIR/redis_memory.csv"
        echo "内存分析完成: $OUTPUT_DIR/redis_memory.csv"
        ;;
    3)
        # 按数据类型导出
        rdb --command json "$RDB_FILE" --type string > "$OUTPUT_DIR/redis_strings.json"
        rdb --command json "$RDB_FILE" --type hash > "$OUTPUT_DIR/redis_hashes.json"
        rdb --command json "$RDB_FILE" --type list > "$OUTPUT_DIR/redis_lists.json"
        rdb --command json "$RDB_FILE" --type set > "$OUTPUT_DIR/redis_sets.json"
        rdb --command json "$RDB_FILE" --type zset > "$OUTPUT_DIR/redis_zsets.json"
        echo "按数据类型导出完成"
        ;;
    *)
        # 默认完整 JSON 导出
        rdb --command json "$RDB_FILE" > "$OUTPUT_DIR/redis_data.json"
        echo "默认 JSON 导出完成: $OUTPUT_DIR/redis_data.json"
        ;;
esac