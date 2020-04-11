import json
import re
import requests
import datetime

today = datetime.date.today().strftime('%Y%m%d')   #今天的日期.json

def crawl_dxy_data():
    """
    爬取丁香园实时统计数据，保存到data目录下，以当前日期作为文件名，存JSON文件
    """
    response = requests.get('https://ncov.dxy.cn/ncovh5/view/pneumonia') #request.get()用于请求目标网站
    print(response.status_code)                                          # 打印状态码


    try:
        url_text = response.content.decode()                             #更推荐使用response.content.deocde()的方式获取响应的html页面
        #print(url_text)
        url_content = re.search(r'window.getAreaStat = (.*?)}]}catch',   #re.search():扫描字符串以查找正则表达式模式产生匹配项的第一个位置 ，然后返回相应的match对象。
                                url_text, re.S)                          #在字符串a中，包含换行符\n，在这种情况下：如果不使用re.S参数，则只在每一行内进行匹配，如果一行没有，就换下一行重新开始;
                                                                         #而使用re.S参数以后，正则表达式会将这个字符串作为一个整体，在整体中进行匹配。
        texts = url_content.group()                                      #获取匹配正则表达式的整体结果
        content = texts.replace('window.getAreaStat = ', '').replace('}catch', '') #去除多余的字符
        json_data = json.loads(content)                                         
        with open('data/' + today + '.json', 'w', encoding='UTF-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
    except:
        print('<Response [%s]>' % response.status_code)


def crawl_statistics_data():
    """
    获取各个省份历史统计数据，保存到data目录下，存JSON文件
    """
    with open('data/'+ today + '.json', 'r', encoding='UTF-8') as file:
        json_array = json.loads(file.read())

    statistics_data = {}
    for province in json_array:
        response = requests.get(province['statisticsData'])
        try:
            statistics_data[province['provinceShortName']] = json.loads(response.content.decode())['data']
        except:
            print('<Response [%s]> for url: [%s]' % (response.status_code, province['statisticsData']))

    with open("data/statistics_data.json", "w", encoding='UTF-8') as f:
        json.dump(statistics_data, f, ensure_ascii=False)


if __name__ == '__main__':
    crawl_dxy_data()
    crawl_statistics_data()