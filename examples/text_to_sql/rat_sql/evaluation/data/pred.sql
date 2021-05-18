1	( select 出版社 from 传记 where 页数 > 400 ) intersect ( select 书名 from 传记 where 出版时间 > "1981-03-24" )	人物传记
2	( select 姓名 from 作者 where 作品数量 >= 60 ) except ( select 姓名 from 作者 order by 出生日期 desc limit 3 )	小说
3	( select 开源课程名称 from 学校的开源课程 order by 课时 desc limit 5 ) except ( select 开源课程名称 from 学校的开源课程 where 主讲教师 != "王建安" )	在线学习平台
4	select avg ( 现价格 ) max ( 原价格 ) from 本月特价书籍	榜单
5	select max ( 电子书售价 ) from 电子书	购书平台
6	select min ( 电子书售价 ) avg ( 购买人数 ) max ( 会员价格 ) from 电子书	购书平台
7	select sum ( 豆瓣评分 ) max ( 1星占比 ) from 书籍	豆瓣读书
8	select 书名 from 传记 where 作者 != "柳润墨" order by 页数 asc	人物传记
9	select 书名, 类型 from 网络小说 where 评分 in ( select max ( 评分 ) from 网络小说 )	网易云阅读
10	select 出版社 from 文集 group by 出版社 order by avg ( 页数 ) desc limit 1	文集
11	select 名称 from 小说改编话剧 where 演出总场次 < ( select max ( 演出总场次 ) from 小说改编话剧 where 演出剧团 != "开心麻花" )	小说
12	select 名称 from 文集 where 页数 < ( select max ( 页数 ) from 文集 where 出版社 != "人民文学出版社" )	文集
13	select 名称 from 文集 where 页数 == ( select max ( 页数 ) from 文集 where 出版社 != "人民文学出版社" )	文集
14	select 名称, 作者 from 书籍 where 豆瓣评分 > 5.4 order by 1星占比 desc	豆瓣读书
15	select 名称, 评价人数 * 1星占比 from 书籍 where 作者 == "塔拉·韦斯特弗"	豆瓣读书
16	select 姓名, 国籍 from 作者 where 作品数量 == ( select max ( 作品数量 ) from 作者 )	小说
17	select 姓名, 逝世日期 - 出生日期 from 作者 where 作品数量 < 50	小说
18	select 讲述朝代 from 中国朝代历史	历史类书籍
