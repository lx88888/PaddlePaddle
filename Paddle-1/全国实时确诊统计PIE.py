import json
import datetime
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# 读原始数据文件
today = datetime.date.today().strftime('%Y%m%d')   #当天日期，例如：20200317
datafile = 'data/'+ today + '.json'

with open(datafile, 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 分析全国实时确诊数据：'confirmedCount'字段
china_data = []
for province in json_array:
    china_data.append((province['provinceShortName'], province['confirmedCount']))
china_data = sorted(china_data, key=lambda x: x[1], reverse=True)                 #reverse=True,表示降序，反之升序


labels = [data[0] for data in china_data]
counts = [data[1] for data in china_data]


p = (
    Pie(init_opts=opts.InitOpts(width="1600px", 
                                height="1200px")
        )
    .add("累计确诊", 
         [list(z) for z in zip(labels, counts)],
         center=["50%", "50%"],#改变位置
         )  
    .set_global_opts(title_opts=opts.TitleOpts(title="全国各省累计确诊统计图",
                                               subtitle='数据来源：丁香园'),
                    legend_opts=opts.LegendOpts(pos_left="91%"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    .render(path='全国实时确诊统计PIE.html')
)