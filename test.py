from pymilvus import connections

try:
    connections.connect(host="47.96.113.144", port="19530")
    print("连接成功")
except Exception as e:
    print(f"连接失败: {e}")