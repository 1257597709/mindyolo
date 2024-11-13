names = [
    'n2005', 's114', 's1547', 's1556', 's1557', 's1607', 's1641', 's1706',
    's1774', 's1778', 's1797', 's1804', 's1805', 's1806', 's1843', 's1853',
    's1854', 's1868', 's1870', 's1877', 's1896', 's1903', 's1906', 's1914',
    's1918', 's1935', 's2011', 's377', 's5562', 's7704', 's8803', 's8804',
    's8813', 's8816', 's8819', 's8832', 's8835', 's8843', 'sn112', 'sn13',
    'sn15', 'sn176', 'sn19', 'sn20', 'sn22', 'sn23'
]

# 生成类别字典
categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]

# 打印结果
import json
print(json.dumps(categories, indent=4))
