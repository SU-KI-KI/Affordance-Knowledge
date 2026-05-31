import json

# 输入文件
input_file = "Output/out_put_202604251058_1-7501.json"

# 输出文件
output_file = "Output/out_put_1-800.json"

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 保留前 800 条（删除最后 38 条）
new_data = data[:800]

# 保存新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"原始数据条数: {len(data)}")
print(f"保留后数据条数: {len(new_data)}")
print(f"已保存到: {output_file}")