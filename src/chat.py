import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("/share2/users/chenxu/code/tools2/biomarker-discovery/tests/mode1/metric_summary.xlsx")

# 将 DataFrame 转换为 HTML 代码
html = df.to_html()

# 将 HTML 代码写入文件
with open("/share2/users/chenxu/code/tools2/biomarker-discovery/tests/mode1/example.html", "w") as f:
    f.write(html)
