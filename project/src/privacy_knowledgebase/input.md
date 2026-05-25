# 个人信息列表

## A. 标识符（Identifiers）

| 信息类型    | 具体示例                              | PIPL | CCPA                   | GDPR | 敏感级别                    | 备注                                    |
| ------- | --------------------------------- | ---- | ---------------------- | ---- | ----------------------- | ------------------------------------- |
| 姓名      | 真实姓名、别名、用户名                       | I    | E                      | E    | 一般                      | CCPA §1798.140(v)(1)(A)；GDPR Art.4(1) |
| 身份证件号码  | 身份证号、社保号、护照号、驾照号                  | I    | E                      | E    | ⚠️（CCPA）/ 一般（GDPR/PIPL） | PIPL通过第四条“可识别性”覆盖；CCPA明确列举为标识符        |
| 联系方式    | 邮政地址、电子邮箱、电话号码                    | I    | E                      | E    | 一般                      | CCPA明确；GDPR列举；PIPL定义覆盖                |
| 在线标识符   | IP地址、Cookie ID、设备ID、移动广告ID、RFID标签 | I    | E                      | E    | 一般                      | GDPR Recital 30明确提及；CCPA明确列举          |
| 账户身份信息  | 账户名、用户名（不含密码）                     | I    | E                      | I    | 一般                      | CCPA作为标识符；PIPL/GDPR可覆盖                |
| ⚠️ 账户凭证 | 登录账号 + 安全码/密码/访问凭证的组合             | –    | E（§1798.140(ae)(1)(B)） | –    | 敏感（CCPA）                | 仅CCPA明确列为敏感信息；单独账号不属此类                |

## B. 敏感个人信息（特殊类别数据）

| 信息类型            | 具体示例                    | PIPL           | CCPA                            | GDPR               | 敏感级别                   | 备注                        |
| --------------- | ----------------------- | -------------- | ------------------------------- | ------------------ | ---------------------- | ------------------------- |
| ⚠️ 生物识别信息       | 指纹、面部识别、虹膜、声纹、DNA（用于识别） | E（§28）         | E（§1798.140(b)）                 | E（Art.4(14)，Art.9） | 敏感                     | 三部法律均明确列举                 |
| ⚠️ 基因数据         | DNA、RNA分析结果，遗传特征        | I（可归入生物识别/健康）  | E（§1798.140(ae)(1)(F)）          | E（Art.4(13)）       | 敏感                     | PIPL未单列，但属于生物识别或健康范畴      |
| ⚠️ 健康信息         | 病历、诊断、治疗记录、残疾、病史        | E（§28）         | E（§1798.140(v)(1)(B)及健康条款）      | E（Art.4(15)，Art.9） | 敏感                     | 三部均明确                     |
| ⚠️ 种族或民族出身      | 种族、民族                   | –              | E（§1798.140(v)(1)(C)）           | E（Art.9）           | 敏感                     | **PIPL未列出**，已删除           |
| ⚠️ 宗教信仰         | 宗教或哲学信仰                 | E（§28）         | E（§1798.140(ae)(1)(D)）          | E（Art.9）           | 敏感                     |                           |
| ⚠️ 政治观点         | 政治立场                    | –              | E（受保护分类）                        | E（Art.9）           | 敏感                     | **PIPL未列出**，已删除           |
| ⚠️ 工会成员身份       | 是否加入工会                  | –              | E（§1798.140(ae)(1)(D)）          | E（Art.9）           | 敏感                     |                           |
| ⚠️ 性取向或性生活      | 性取向、性生活信息               | –              | E（§1798.140(ae)(2)(C)）          | E（Art.9）           | 敏感                     |                           |
| ⚠️ 金融账户信息       | 银行账号、信用卡号、征信信息          | E（§28 金融账户）    | E（§1798.140(v)(1)(D)商业信息，亦作为敏感） | I                  | 敏感（PIPL/CCPA）          | GDPR未单列为敏感，但可视为个人数据       |
| ⚠️ 精确地理位置       | 半径≤1850英尺的地理位置          | E（行踪轨迹）        | E（§1798.140(ae)(1)(C)）          | E（位置数据，Art.4(1)）   | 敏感                     |                           |
| ⚠️ 行踪轨迹         | 实时位置、出行记录               | E（§28）         | I                               | I                  | 敏感                     | CCPA归入地理位置数据；GDPR归入位置数据   |
| ⚠️ 不满XX周岁未成年人信息 | 儿童个人信息                  | E（§28，14周岁以下）  | E（§1798.120，13/16周岁）            | E（Art.8，16/13周岁）   | 敏感                     | 年龄门槛不同                    |
| ⚠️ 私人通信内容       | 邮件、短信、即时通讯内容            | I（隐私权保护）       | E（§1798.140(ae)(1)(E)）          | I（定义覆盖）            | 敏感（CCPA）；一般（GDPR/PIPL） | CCPA明确为敏感；GDPR无特殊分类但属个人数据 |
| ⚠️ 特定身份         | 法律未穷尽定义                 | E（§28 列举，但无示例） | –                               | –                  | 敏感                     | 仅PIPL提及，不应随意举例            |

## C. 数字活动信息（Digital Activity）

| 信息类型 | 具体示例            | PIPL | CCPA | GDPR | 敏感级别 | 备注                         |
| ---- | --------------- | ---- | ---- | ---- | ---- | -------------------------- |
| 浏览历史 | 访问的网站、网页浏览记录    | I    | E    | I    | 一般   | CCPA明确（§1798.140(v)(1)(F)） |
| 搜索历史 | 搜索引擎查询记录        | I    | E    | I    | 一般   | 同上                         |
| 交互信息 | 与网站、应用、广告的点击、停留 | I    | E    | I    | 一般   | 同上                         |
| 设备信息 | 设备类型、操作系统、MAC地址 | I    | I    | I    | 一般   | 均可通过定义覆盖                   |

## D. 商业与职业信息（Commercial & Professional）

| 信息类型    | 具体示例                 | PIPL | CCPA                  | GDPR | 敏感级别 | 备注                         |
| ------- | -------------------- | ---- | --------------------- | ---- | ---- | -------------------------- |
| 购买记录    | 购买或考虑购买的产品或服务        | I    | E                     | I    | 一般   | CCPA明确列举商业信息               |
| 消费历史或倾向 | 消费习惯、购物偏好            | I    | E                     | I    | 一般   | 同上                         |
| 个人财产记录  | 房产、车辆、贵重物品           | I    | E                     | I    | 一般   | CCPA明确                     |
| 教育信息    | 学历、成绩单、学生记录（FERPA定义） | –    | E（§1798.140(v)(1)(J)） | I    | 一般   | PIPL无此类别；GDPR通过定义覆盖        |
| 职业信息    | 工作单位、职位、工作经历         | I    | E                     | I    | 一般   | CCPA明确（§1798.140(v)(1)(I)） |
| 专业资质    | 专业资格、执照、证书           | I    | I                     | I    | 一般   | 无法律明确列举，均通过定义覆盖            |

## E. 生理与感官信息（Physical & Sensory）

| 信息类型 | 具体示例         | PIPL | CCPA | GDPR | 敏感级别               | 备注                         |
| ---- | ------------ | ---- | ---- | ---- | ------------------ | -------------------------- |
| 身体特征 | 身高、体重、血型     | I    | I    | I    | 一般                 | 均可由“生理因素”覆盖                |
| 感官信息 | 照片、录音、录像、热成像 | I    | E    | I    | 一般（照片仅在用于生物识别时为敏感） | CCPA明确（§1798.140(v)(1)(H)） |

## F. 画像与推论（Profiling & Inferences）

| 信息类型 | 具体示例             | PIPL        | CCPA                  | GDPR        | 敏感级别 | 备注                        |
| ---- | ---------------- | ----------- | --------------------- | ----------- | ---- | ------------------------- |
| 用户画像 | 偏好、特征、心理趋势、行为、态度 | I（§24自动化决策） | E（§1798.140(v)(1)(K)） | E（Art.4(4)） | 一般   | CCPA/GDPR明确；PIPL通过自动化决策覆盖 |
| 行为预测 | 工作表现、经济状况、健康趋势等  | I           | E（推论）                 | E（Art.4(4)） | 一般   | GDPR明确列举预测内容              |

## G. 家庭信息（Household Information）

| 信息类型   | 具体示例           | PIPL | CCPA                            | GDPR | 敏感级别 | 备注                          |
| ------ | -------------- | ---- | ------------------------------- | ---- | ---- | --------------------------- |
| 家庭成员信息 | 婚姻状况、子女信息、同住人员 | I    | E（§1798.140(v)(1) 包含 household） | I    | 一般   | CCPA明确将“家庭”纳入；PIPL不单独列出但可覆盖 |

## H. 特殊处理状态（Special Processing States）

| 状态                   | 说明            | 是否属于个人信息（PIPL） | 是否属于个人信息（CCPA）      | 是否属于个人信息（GDPR）   | 备注                      |
| -------------------- | ------------- | -------------- | ------------------- | ---------------- | ----------------------- |
| 假名化数据（Pseudonymized） | 需额外信息才能识别     | I（视为个人信息）      | I（视为个人信息）           | E（明确属于，Art.4(5)） | GDPR明确，其他法律无相反规定        |
| 去标识化数据（Deidentified） | 不能直接识别但可能重新识别 | –（无统一概念）       | E（可排除，§1798.140(m)） | –                | CCPA有条件排除；PIPL“去标识化”不等同 |
| 匿名化信息（Anonymized）    | 无法识别且不能复原     | E（排除，§4）       | E（排除）               | E（排除，Recital 26） | 均不属个人信息                 |
| 聚合消费者信息              | 群体统计，个体已移除    | I（可视为匿名）       | E（排除，§1798.140(b)）  | I（可视为匿名）         | CCPA明确排除                |

---

## 附录：法律参考文件自动摘录

> 由 `legal_pdf_ingest` 于 2026-05-24T06:31:42Z UTC 从 `法律参考文件/` 生成，供人工对照；
> **不自动改写** 上文 A–H 矩阵表格。更新矩阵请人工编辑后运行 `kb-build`。

### 来源：CCPA.pdf

Full Text of the CCPA 
includes CPRA changes

California Consumer Privacy Act  www.teachprivacy.com 
2 
 
CALIFORNIA CONSUMER PRIVACY ACT (CCPA) 
updated text with CPRA changes 
from the official website 
CIVIL CODE - CIV 
DIVISION 3. OBLIGATIONS [1427 - 3273.16] 
  ( Heading of Division 3 amended by Stats. 1988, Ch. 160, Sec. 14. ) 
PART 4. OBLIGATIONS ARISING FROM PARTICULAR TRANSACTIONS [1738 - 3273.16] 
  ( Part 4 enacted 1872. ) 
TITLE 1.81.5. California Consumer Privacy Act of 2018 [1798.100 - 1798.199.100] 
  ( Title 1.81.5 added by Stats. 2018, Ch. 55, Sec. 3. ) 
 
1798.100. 
(a) A consumer shall have the right to request that a business that collects a consumer’s personal 
information disclose to that consumer the categories and specific pieces of personal information the 
business has collected. 
(b) A business that collects a consumer’s personal information shall, at or before the point of 
collection, inform consumers as to the categories of personal information to be collected and the 
purposes for which the categories of personal information shall be used. A business shall not collect 
additional categories of personal information or use personal information collected for additional 
purposes without providing the consumer with notice consistent with this section. 
(c) A business shall provide the information specified in subdivision (a) to a consumer only upon 
receipt of a verifiable consumer request. 
(d) A business that receives a verifiable consumer request from a consumer to access personal 
information shall promptly take steps to disclose and deliver, free of charge to the consumer, the 
personal information required by this section. The information may be delivered by mail or 
electronically, and if provided electronically, the information shall be in a portable and, to the extent 
technically feasible, readily useable format that allows the consumer to transmit this information to 
another entity without hindrance. A business may provide personal information to a consumer at any 
time, but shall not be required to provide personal information to a consumer more than twice in a 
12-month period.

California Consumer Privacy Act  www.teachprivacy.com 
3 
 
(e) This section shall not require a business to retain any personal information collected for a single, 
one-time transaction, if such information is not sold or retained by the business or to reidentify or 
otherwise link information that is not maintained in a manner that would be considered personal 
information. 
(Amended by Stats. 2019, Ch. 757, Sec. 1. (AB 1355) Effective January 1, 2020. Superseded on January 1, 2023; 
see amendment by Proposition 24.) 
1798.100. 
General Duties of Businesses that Collect Personal Information 
(a) A business that controls the collection of a consumer’s personal information shall, at or before the 
point of collection, inform consumers of the following: 
(1) The categories of personal information to be collected and the purposes for which the 
categories of personal information are collected or used and whether that information is sold or 
shared. A business shall not collect additional categories of personal information or use personal 
information collected for additional purposes that are incompatible with the disclosed purpose for 
which the personal information was collected without providing the consumer with notice 
consistent with this section. 
(2) If the business collects sensitive personal information, the categories of sensitive personal 
information to be collected and the purposes for which the categories of sensitive personal 
information are collected or used, and whether that information is sold or shared. A business shall 
not collect additional categories of sensitive personal information or use sensitive personal 
information collected for additional purposes that are incompatible with the disclosed purpose for 
which the sensitive personal information was collected without providing the consumer with 
notice consistent with this section. 
(3) The length of time the business intends to retain each category of personal information, 
including sensitive personal information, or if that is not possible, the criteria used to determine 
that period provided that a business shall not retain a consumer’s personal information or sensitive 
personal information for each disclosed purpose for which the personal information was collected 
for longer than is reasonably necessary for that disclosed purpose. 
(b) A business that, acting as a third party, controls the collection of personal information about a 
consumer may satisfy its obligation under subdivision (a) by providing the required information 
prominently and conspicuously on the homepage of its internet website. In addition, if a business 
acting as a third party controls the collection of personal information about a consumer on its 
premises, including in a vehicle, then the business shall, at or before the point of collection, inform 
consumers as to the categories of personal information to be collected and the purposes for which 
the categories of personal information are used, and whether that personal information is sold, in a 
clear and conspicuous manner at the location.

California Consumer Privacy Act  www.teachprivacy.com 
4 
 
(c) A business’ collection, use, retention, and sharing of a consumer’s personal information shall be 
reasonably necessary and proportionate to achieve the purposes for which the personal information 
was collected or processed, or for another disclosed purpose that is compatible with the context in 
which the personal information was collected, and not further processed in a manner that is 
incompatible with those purposes. 
(d) A business that collects a consumer’s personal information and that sells that personal 
information to, or shares it with, a third party or that discloses it to a service provider or contractor 
for a business purpose shall enter into an agreement with the third party, service provider, or 
contractor, that: 
(1) Specifies that the personal information is sold or disclosed by the business only for limited and 
specified purposes. 
(2) Obligates the third party, service provider, or contractor to comply with applicable obligations 
under this title and obligate those persons to provide the same level of privacy protection as is 
required by this title. 
(3) Grants the business rights to take reasonable and appropriate steps to help ensure that the 
third party, service provider, or contractor uses the personal information transferred in a manner 
consistent with the business’ obligations under this title. 
(4) Requires the third party, service provider, or contractor to notify the business if it makes a 
determination that it can no longer meet its obligations under this title. 
(5) Grants the business the right, upon notice, including under paragraph (4), to take reasonable 
and appropriate steps to stop and remediate unauthorized use of personal information. 
(e) A business that collects a consumer’s personal information shall implement reasonable security 
procedures and practices appropriate to the nature of the personal information to protect the 
personal information from unauthorized or illegal access, destruction, use, modification, or disclosure 
in accordance with Section 1798.81.5. 
(f) Nothing in this section shall require a business to disclose trade secrets, as specified in regulations 
adopted pursuant to paragraph (3) of subdivision (a) of Section 1798.185. 
(Amended November 3, 2020, by initiative Proposition 24, Sec. 4. Effective December 16, 2020. Operative 
January 1, 2023, pursuant to Sec. 31 of Proposition 24.) 
1798.105. 
(a) A consumer shall have the right to request that a business delete any personal information about 
the consumer which the business has collected from the consumer. 
(b) A business that collects personal information about consumers shall disclose, pursuant to Section 
1798.130, the consumer’s rights to request the deletion of the consumer’s personal information.

California Consumer Privacy Act  www.teachprivacy.com 
5 
 
(c) A business that receives a verifiable consumer request from a consumer to delete the consumer’s 
personal information pursuant to subdivision (a) of this section shall delete the consumer’s personal 
information from its records and direct any service providers to delete the consumer’s personal 
information from their records. 
(d) A business or a service provider shall not be required to comply with a consumer’s request to 
delete the consumer’s personal information if it is necessary for the business or service provider to 
maintain the consumer’s personal information in order to: 
(1) Complete the transaction for which the personal information was collected, fulfill the terms of a 
written warranty or product recall conducted in accordance with federal law, provide a good or 
service requested by the consumer, or reasonably anticipated within the context of a business’ 
ongoing business relationship with the consumer, or otherwise perform a contract between the 
business and the consumer. 
(2) Detect security incidents, protect against malicious, deceptive, fraudulent, or illegal activity; or 
prosecute those responsible for that activity. 
(3) Debug to identify and repair errors that impair existing intended functionality. 
(4) Exercise free speech, ensure the right of another consumer to exercise that consumer’s right of 
free speech, or exercise another right provided for by law. 
(5) Comply with the California Electronic Communications Privacy Act pursuant to Chapter 3.6 
(commencing with Section 1546) of Title 12 of Part 2 of the Penal Code. 
(6) Engage in public or peer-reviewed scientific, historical, or statistical research in the public 
interest that adheres to all other applicable ethics and privacy laws, when the business’ deletion of 
the information is likely to render impossible or seriously impair the achievement of such research, 
if the consumer has provided informed consent. 
(7) To enable solely internal uses that are reasonably aligned with the expectations of the 
consumer based on the consumer’s relationship with the business. 
(8) Comply with a legal obligation. 
(9) Otherwise use the consumer’s personal information, internally, in a lawful manner that is 
compatible with the context in which the consumer provided the information. 
(Amended by Stats. 2019, Ch. 751, Sec. 1. (AB 1146) Effective January 1, 2020. Superseded on January 1, 2023; 
see amendment by Proposition 24.)

California Consumer Privacy Act  www.teachprivacy.com 
6 
 
1798.105. 
Consumers’ Right to Delete Personal Information 
(a) A consumer shall have the right to request that a business delete any personal information about 
the consumer which the business has collected from the consumer. 
(b) A business that collects personal information about consumers shall disclose, pursuant to Section 
1798.130, the consumer’s rights to request the deletion of the consumer’s personal information. 
(c) (1) A business that receives a verifiable consumer request from a consumer to delete the 
consumer’s personal information pursuant to subdivision (a) of this section shall delete the 
consumer’s personal information from its records, notify any service providers or contractors to 
delete the consumer’s personal information from their records, and notify all third parties to whom 
the business has sold or shared the personal information to delete the consumer’s personal 
information unless this proves impossible or involves disproportionate effort. 
(2) The business may maintain a confidential record of deletion requests solely for the purpose of 
preventing the personal information of a consumer who has submitted a deletion request from 
being sold, for compliance with laws or for other purposes, solely to the extent permissible under 
this title. 
(3) A service provider or contract

…（摘录已截断）

### 来源：GDPR.pdf

I 
(Legislativ e acts) 
REGUL A TIONS 
REGUL A TION (EU) 2016/679 OF THE EUR OPEAN P ARLIAMENT AND OF THE COUNCIL 
of 27 Apr il 2016 
on the protection of natural persons with regard to the processing of personal data and on the free 
mo v ement of such dat a, and repealing Directiv e 95/46/EC (General Data Protection Regulation) 
(T ext with EEA relevance) 
THE EUR OPEAN P ARLIAMENT AND THE COUNCIL OF THE EUR OPEAN UNION, 
Ha ving regar d to the T reaty on the Functioning of the European Union, and in par ticular Ar ticle 16 thereof, 
Ha ving regar d to the proposal from the European Commission, 
Af ter transmission of the draf t legislative act to the national parliaments, 
Ha ving regar d to the opinion of the European Economic and Social Committ ee (
1
), 
Ha ving regar d to the opinion of the Committ ee of the Regions (
2
), 
A cting in accordance with the ordinar y legislative procedure (
3
), 
Whereas: 
(1)  The protection of natural persons in relation to the processing of personal data is a fundamental r ight. 
Ar ticle 8(1) of the Char ter of Fundamental Rights of the European Union (the ‘Char te r ’) and Ar ticle 16(1) of the 
T reaty on the Functioning of the European U nion (TFEU) pro vide that ever y one has the r ight to the prot ection of 
personal data concer ning him or her . 
(2)  The pr inciples of, and r ules on the protection of natural persons with regard to the processing of their personal 
data should, whate ver their nationality or residence, respect their fundamental r ights and freedoms, in par ticular 
their r ight to the prote ction of personal data. This Regulation is inte nded to contr ibute to the accompl ishment of 
an area of freedom, secur ity and justice and of an economic union, to economic and social progress, to the 
strengthening and the converg ence of the economies within the inte r nal mark et, and to the well-being of natural 
persons. 
(3)  Directive 95/46/EC of the European P arliament and of the Council (
4
) seeks to har monise the prot ection of 
fundamental r ights and freedoms of natural persons in respect of processing activities and to ensure the free f lo w 
of personal data between Member State s. 
4.5.2016 L 119/1 Official Jour nal of the European Union EN     
(
1
) OJ C 229, 31.7.2012, p. 90. 
(
2
) OJ C 391, 18.12.2012, p. 127. 
(
3
) P osition of the European P arliament of 12 March 2014 (not yet published in the Offi cial Jour nal) and position of the Council at first 
reading of 8 Apr il 2016 (not yet published in the Offi cial Jour nal). P osition of the European Parliament of 14 Apr il 2016. 
(
4
) Directive 95/46/EC of the European Parliament and of the Council of 24 October 1995 on the protection of individuals with regard t o 
the processing of personal data and on the free mo vement of such data (OJ L 281, 23.11.1995, p. 31).

(4)  The processing of personal data should be designed to ser ve mankind. The r ight to the prote ction of personal 
data is not an absolute r ight ; it must be considered in relation to its function in society and be balanced ag ainst 
other fundamental r ights, in accordance with the pr inciple of propor tionality . This Regulation respects all 
fundamental r ights and obser ves the freedoms and pr inciples recognised in the Char ter as enshr ined in the 
T reaties, in par ticular the respect f or pr ivate and f amily life, home and communications, the protection of 
personal data, freedom of thought, conscience and religion, freedom of expression and inf or mation, freedom to 
conduct a business, the r ight to an eff ective remedy and to a fa ir tr ial, and cultural, religious and linguistic 
diversity . 
(5)  The economic and social inte gration resulting from the functioning of the intern al mark et has led to a substantial 
increase in cross-border f lo ws of personal data. The exc hange of personal data between public and pr ivate actor s, 
including natural persons, associations and under takings across the U nion has increased. National author ities in 
the Member State s are being called upon by Union law to cooperate and exc hange personal data so as to be able 
to perform their duties or car r y out tasks on behalf of an author ity in another Member State . 
(6)  R apid tec hnological developments and g lobalisation have brought new c hallenges f or the protection of personal 
data. The scale of the collection and shar ing of personal data has increased signif icantly . T echn ology allo ws both 
pr ivate companies and public author ities to mak e use of personal data on an un precedented scale in order to 
pursue their activities. Natural persons increasing ly mak e personal inf or mation available publicly and g lobally . 
T ech nology has transf or med both the econom y and social life, and should fur ther facilitat e the free f lo w of 
personal data within the U nion and the transfer to third countr ies and inte r national orga nisations, while ensur ing 
a high level of the protection of personal data. 
(7)  Those developments require a strong and more coherent data prot ection framew ork in the U nion, back ed by 
strong enf orcement, given the impor tance of creating the tr ust that will allo w the digital econom y to develop 
across the internal mark et. Natural persons should hav e control of their o wn personal data. Legal and practical 
cer tainty f or natural persons, economic operat ors and public author ities should be enhanced. 
(8)  Where this Regulation pro vides f or specific ations or restr ictions of its r ules by Member Stat e la w , Member Stat es 
ma y , as f ar as necessar y f or coherence and f or making the national pro visions compreh ensible to the persons to 
whom they apply , incor porate elements of this Regulation into their national la w . 
(9)  The objectives and pr inciples of Directive 95/46/EC remain sound, but it has not prevente d fragmentation in the 
im plementation of data protection across the Union, legal uncer tainty or a widespread public percep tion that 
there are signif icant r isks to the protection of natural persons, in par ticular with regar d to online activity . 
Diff erences in the level of prote ction of the r ights and freedoms of natural persons, in par ticular the r ight to the 
prot ection of personal data, with regard to the processing of personal data in the Member Stat es ma y prevent the 
free f lo w of personal data throughout the U nion. Those diffe rences ma y theref ore constitute an obstacle to the 
pursuit of economic activities at the level of the Union, distor t competition and impede author ities in the 
disc harge of their responsibilities under U nion law . Such a differ ence in levels of protection is due to the 
existen ce of differ ences in the im plementation and application of Directive 95/46/EC. 
(10)  In order to ensure a consistent and high level of prot ection of natural persons and to remo ve the obstacles to 
f lo ws of personal data within the Union, the level of prot ection of the r ights and freedoms of natural persons 
with regar d to the processing of such data should be equivalent in all Member Stat es. Consisten t and 
homog enous application of the r ules f or the prote ction of the fundamental r ights and freedoms of natural 
persons with regard to the processing of personal data should be ensured throughout the Uni on. Reg arding the 
processing of personal data f or compl iance with a legal oblig ation, f or the performa nce of a task car r ied out in 
the public inte rest or in the exer cise of offici al author ity vested in the controller , Member States should be 
allo wed to maintain or introduce national pro visions to fur ther specify the application of the r ules of this 
Regulation. In conjunction with the ge neral and hor izontal la w on data protection im plementing Directive 
95/46/EC, Member State s hav e several sect or -specif ic laws in areas that need more specifi c pro visions. This 
Regulation also pro vides a margin of manoeuvre f or Member Stat es to specify its r ules, including f or the 
processing of special categor ies of personal data (‘sensitive data’). T o that exte nt, this Regulation does not excl ude 
Member Stat e law that sets out the circumstances f or specifi c processing situations, including det er mining more 
precisely the conditions under which the processing of personal data is la wful. 
4.5.2016 L 119/2 Official Jour nal of the European Union EN

(11)  Effe ctive prote ction of personal data throughout the Union requires the strengthening and setting out in detail of 
the r ights of data subjects and the oblig ations of those who process and det er mine the processing of personal 
data, as well as equivalent po wers f or monitori ng and ensur ing compl iance with the r ules f or the prote ction of 
personal data and equivalent sanctions f or infr ing ements in the Member Stat es. 
(12)  Ar ticle 16(2) TFEU mandates the European P arliament and the Council to la y do wn the r ules relating to the 
prot ection of natural persons with regard to the processing of personal data and the r ules relating to the free 
mo vement of personal data. 
(13)  In order to ensure a consistent level of protection f or natural persons throughout the Union and to prevent 
diverg ences hamp er ing the free mo vement of personal data within the inte r nal mark et, a Regulation is necessar y 
to pro vide leg al cer tainty and transparency f or economic operat ors, including micro, small and medium-sized 
ent er pr ises, and to pro vide natural persons in all Member States with the same level of lega lly enf orceable r ights 
and obligations and responsibilities f or controllers and processors, to ensure consistent monitoring of the 
processing of personal data, and equivalent sanctions in all Member Stat es as well as eff ective cooperation 
between the super visor y author ities of diffe rent Member States. The proper functioning of the internal mark et 
requires that the free mo vement of personal data within the U nion is not restr ict ed or prohibit ed f or reasons 
connecte d with the protection of natural persons with regard to the processing of personal data. T o take account 
of the specific situation of micro, small and medium-sized ent er pr ises, this Regulation includes a derog ation f or 
org anisations with f ewer than 250 emplo y ees with regar d to record-keeping. In addition, the U nion institutions 
and bodies, and Member Stat es and their super visor y author ities, are encourag ed to tak e account of the specifi c 
needs of micro, small and medium-sized enterpr ises in the application of this Regulation. The notion of micro, 
small and medium-sized ent er pr ises should dra w from Ar ticle 2 of the Annex to Commission Recommendation 
2003/361/EC (
1
). 
(14)  The protection aff orded by this Regulation should apply to natural persons, whatever their nationality or place of 
residence, in relation to the processing of their personal data. This Regulation does not co ver the processing of 
personal data which concer ns legal persons and in par ticular under takings established as leg al persons, including 
the name and the f or m of the leg al person and the contact details of the leg al person. 
(15) In order to prevent creating a ser ious r isk of circumvention, the prote ction of natural persons should be tec hnolo ­
gically neutral and should not depend on the te chniques used. The prot ection of natural persons should apply to 
the processing of personal data by automat ed means, as well as to manual processing, if the personal data are 
contained or are inte nded to be contained in a f iling system. Files or sets of files, as well as their co ver pag es, 
whic h are not str uctured according to specifi c cr it er ia should not f all within the scope of this Regulation. 
(16)  This Regulation does not apply to issues of prot ection of fundamental r ights and freedoms or the free f lo w of 
personal data relat ed to activities whic

…（摘录已截断）

### 来源：个人信息保护法.pdf

中华人民共和国个人信息保护法 
（2021 年 8 月 20 日第十三届全国人民代表大会常务委员会
第三十次会议通过） 
 
目 录 
第一章 总 则 
第二章 个人信息处理规则 
第一节 一般规定 
第二节 敏感个人信息的处理规则 
第三节 国家机关处理个人信息的特别规定 
第三章 个人信息跨境提供的规则 
第四章 个人在个人信息处理活动中的权利 
第五章 个人信息处理者的义务 
第六章 履行个人信息保护职责的部门 
第七章 法律责任 
第八章 附 则 
第一章 总 则 
第一条 为了保护个人信息权益，规范个人信息处理活
动，促进个人信息合理利用，根据宪法，制定本法。 
第二条 自然人的个人信息受法律保护， 任何组织、 个人

不得侵害自然人的个人信息权益。 
第三条  在中华人民共和国境内处理自然人个人信息的
活动，适用本法。 
在中华人民共和国境外处理中华人民共和国境内自然
人个人信息的活动，有下列情形之一的，也适用本法： 
（一）以向境内自然人提供产品或者服务为目的； 
（二）分析、评估境内自然人的行为； 
（三）法律、行政法规规定的其他情形。 
第四条  个人信息是以电子或者其他方式记录的与已识
别或者可识别的自然人有关的各种信息，不包括匿名化处理
后的信息。 
个人信息的处理包括个人信息的收集、存储、使用、加
工、传输、提供、公开、删除等。 
第五条 处理个人信息应当遵循合法、 正当、 必要和诚信
原则，不得通过误导、欺诈、胁迫等方式处理个人信息。 
第六条 处理个人信息应当具有明确、 合理的目的， 并应
当与处理目的直接相关，采取对个人权益影响最小的方式。 
收集个人信息，应当限于实现处理目的的最小范围，不
得过度收集个人信息。 
第七条 处理个人信息应当遵循公开、 透明原则， 公开个
人信息处理规则，明示处理的目的、方式和范围。 
第八条  处理个人信息应当保证个人信息的质量，避免
因个人信息不准确、不完整对个人权益造成不利影响。

第九条 个人信息处理者应当对其个人信息处理活动负
责，并采取必要措施保障所处理的个人信息的安全。 
第十条 任何组织、 个人不得非法收集、 使用、 加工、 传
输他人个人信息，不得非法买卖、提供或者公开他人个人信
息； 不得从事危害国家安全、 公共利益的个人信息处理活动。 
第十一条  国家建立健全个人信息保护制度，预防和惩
治侵害个人信息权益的行为 ，加强个人信息保护宣传教育，
推动形成政府、企业、相关社会组织、公众共同参与个人信
息保护的良好环境。 
第十二条  国家积极参与个人信息保护国际规则的制定，
促进个人信息保护方面的国际交流与合作，推动与其他国家、
地区、国际组织之间的个人信息保护规则、标准等互认。 
第二章 个人信息处理规则 
第一节 一般规定 
第十三条  符合下列情形之一的，个人信息处理者方可
处理个人信息： 
（一）取得个人的同意； 
（二） 为订立、 履行个人作为一方当事人的合同所必需，
或者按照依法制定的劳动规章制度和依法签订的集体合同
实施人力资源管理所必需； 
（三）为履行法定职责或者法定义务所必需； 
（四）为应对突发公共卫生事件，或者紧急情况下为保

护自然人的生命健康和财产安全所必需； 
（五）为公共利益实施新闻报道、舆论监督等行为，在
合理的范围内处理个人信息； 
（六）依照本法规定在合理的范围内处理个人自行公开
或者其他已经合法公开的个人信息； 
（七）法律、行政法规规定的其他情形。 
依照本法其他有关规定，处理个人信息应当取得个人同
意，但是有前款第二项至第七项规定情形的，不需取得个人
同意。 
第十四条  基于个人同意处理个人信息的，该同意应当
由个人在充分知情的前提下自愿、明确作出。法律、行政法
规规定处理个人信息应当取得个人单独同意或者书面同意
的，从其规定。 
个人信息的处理目的、处理方式和处理的个人信息种类
发生变更的，应当重新取得个人同意。 
第十五条  基于个人同意处理个人信息的，个人有权撤
回其同意。个人信息处理者应当提供便捷的撤回同意的方式。 
个人撤回同意，不影响撤回前基于个人同意已进行的个
人信息处理活动的效力。 
第十六条  个人信息处理者不得以个人不同意处理其个
人信息或者撤回同意为由，拒绝提供产品或者服务；处理个
人信息属于提供产品或者服务所必需的除外。 
第十七条  个人信息处理者在处理个人信息前，应当以

显著方式、清晰易懂的语言真实、准确、完整地向个人告知
下列事项： 
（一）个人信息处理者的名称或者姓名和联系方式； 
（二）个人信息的处理目的、处理方式，处理的个人信
息种类、保存期限； 
（三）个人行使本法规定权利的方式和程序； 
（四）法律、行政法规规定应当告知的其他事项。 
前款规定事项发生变更的，应当将变更部分告知个人。 
个人信息处理者通过制定个人信息处理规则的方式告
知第一款规定事项的，处理规则应当公开，并且便于查阅和
保存。 
第十八条 个人信息处理者处理个人信息， 有法律、 行政
法规规定应当保密或者不需要告知的情形的，可以不向个人
告知前条第一款规定的事项。 
紧急情况下为保护自然人的生命健康和财产安全无法
及时向个人告知的，个人信息处理者应当在紧急情况消除后
及时告知。 
第十九条 除法律、 行政法规另有规定外， 个人信息的保
存期限应当为实现处理目的所必要的最短时间。 
第二十条  两个以上的个人信息处理者共同决定个人信
息的处理目的和处理方式的 ，应当约定各自的权利和义务。
但是，该约定不影响个人向其中任何一个个人信息处理者要
求行使本法规定的权利。

个人信息处理者共同处理个人信息，侵害个人信息权益
造成损害的，应当依法承担连带责任。 
第二十一条  个人信息处理者委托处理个人信息的，应
当与受托人约定委托处理的目的、期限、处理方式、个人信
息的种类、保护措施以及双方的权利和义务等，并对受托人
的个人信息处理活动进行监督。 
受托人应当按照约定处理个人信息，不得超出约定的处
理目的、 处理方式等处理个人信息； 委托合同不生效、 无效、
被撤销或者终止的，受托人应当将个人信息返还个人信息处
理者或者予以删除，不得保留。 
未经个人信息处理者同意，受托人不得转委托他人处理
个人信息。 
第二十二条 个人信息处理者因合并、 分立、 解散、 被宣
告破产等原因需要转移个人信息的，应当向个人告知接收方
的名称或者姓名和联系方式。接收方应当继续履行个人信息
处理者的义务。接收方变更原先的处理目的、处理方式的，
应当依照本法规定重新取得个人同意。 
第二十三条  个人信息处理者向其他个人信息处理者提
供其处理的个人信息的，应当向个人告知接收方的名称或者
姓名、联系方式、处理目的、处理方式和个人信息的种类，
并取得个人的单独同意。接收方应当在上述处理目的、处理
方式和个人信息的种类等范围内处理个人信息。接收方变更
原先的处理目的、处理方式的，应当依照本法规定重新取得

个人同意。 
第二十四条  个人信息处理者利用个人信息进行自动化
决策，应当保证决策的透明度和结果公平、公正，不得对个
人在交易价格等交易条件上实行不合理的差别待遇。 
通过自动化决策方式向个人进行信息推送、商业营销，
应当同时提供不针对其个人特征的选项，或者向个人提供便
捷的拒绝方式。 
通过自动化决策方式作出对个人权益有重大影响的决
定，个人有权要求个人信息处理者予以说明，并有权拒绝个
人信息处理者仅通过自动化决策的方式作出决定。 
第二十五条  个人信息处理者不得公开其处理的个人信
息，取得个人单独同意的除外。 
第二十六条  在公共场所安装图像采集、个人身份识别
设备，应当为维护公共安全所必需，遵守国家有关规定，并
设置显著的提示标识。所收集的个人图像、身份识别信息只
能用于维护公共安全的目的，不得用于其他目的；取得个人
单独同意的除外。 
第二十七 条 个人信息处理者可以在合理的范围内处理
个人自行公开或者其他已经合法公开的个人信息；个人明确
拒绝的除外。个人信息处理者处理已公开的个人信息，对个
人权益有重大影响的，应当依照本法规定取得个人同意。

第二节 敏感个人信息的处理规则 
第二十八条  敏感个人信息是一旦泄露或者非法使用，
容易导致自然人的人格尊严受到侵害或者人身、财产安全受
到危害的个人信息，包括生物识别、宗教信仰、特定身份、
医疗健康、金融账户、行踪轨迹等信息，以及不满十四周岁
未成年人的个人信息。 
只有在具有特定的目的和充分的必要性，并采取严格保
护措施的情形下，个人信息处理者方可处理敏感个人信息。 
第二十九条  处理敏感个人信息应当取得个人的单独同
意；法律、行政法规规定处理敏感个人信息应当取得书面同
意的，从其规定。 
第三十条  个人信息处理者处理敏感个人信息的，除本
法第十七条第一款规定的事项外，还应当向个人告知处理敏
感个人信息的必要性以及对个人权益的影响；依照本法规定
可以不向个人告知的除外。 
第三十一条  个人信息处理者处理不满十四周岁未成年
人个人信息的，应当取得未成年人的父母或者其他监护人的
同意。 
个人信息处理者处理不满十四周岁未成年人个人信息
的，应当制定专门的个人信息处理规则。 
第三十二条  法律、行政法规对处理敏感个人信息规定
应当取得相关行政许可或者作出其他限制的，从其规定。

第三节 国家机关处理个人信息的特别规定 
第三十三条 国家机关处理个人信息的活动，适用本法；
本节有特别规定的，适用本节规定。 
第三十四条  国家机关为履行法定职责处理个人信息，
应当依照法律、行政法规规定的权限、程序进行，不得超出
履行法定职责所必需的范围和限度。 
第三十五条  国家机关为履行法定职责处理个人信息，
应当依照本法规定履行告知义务；有本法第十八条第一款规
定的情形，或者告知将妨碍国家机关履行法定职责的除外。 
第三十六条  国家机关处理的个人信息应当在中华人民
共和国境内存储；确需向境外提供的，应当进行安全评估。
安全评估可以要求有关部门提供支持与协助。 
第三十七条  法律、法规授权的具有管理公共事务职能
的组织为履行法定职责处理个人信息，适用本法关于国家机
关处理个人信息的规定。 
第三章 个人信息跨境提供的规则 
第三十八条  个人信息处理者因业务等需要，确需向中
华人民共和国境外提供个人信息的，应当具备下列条件之一： 
（一）依照本法第四十条的规定通过国家网信部门组织
的安全评估； 
（二）按照国家网信部门的规定经专业机构进行个人信
息保护认证；

（三）按照国家网信部门制定的标准合同与境外接收方
订立合同，约定双方的权利和义务； 
（四）法律、行政法规或者国家网信部门规定的其他条
件。 
中华人民共和国缔结或者参加的国际条约、协定对向中
华人民共和国境外提供个人信息的条件等有规定的，可以按
照其规定执行。 
个人信息处理者应当采取必要措施，保障境外接收方处
理个人信息的活动达到本法规定的个人信息保护标准。 
第三十九条 个人信息处理者向中华人民共和国境外提
供个人信息的，应当向个人告知境外接收方的名称或者姓名、
联系方式、处理目的、处理方式、个人信息的种类以及个人
向境外接收方行使本法规定权利的方式和程序等事项，并取
得个人的单独同意。 
第四十条  关键信息基础设施运营者和处理个人信息达
到国家网信部门规定数量的个人信息处理者，应当将在中华
人民共和国境内收集和产生的个人信息存储在境内。确需向
境外提供的， 应当通过国家网信部门组织的安全评估； 法律、
行政法规和国家网信部门规定可以不进行安全评估的，从其
规定。 
第四十一条  中华人民共和国 主管机关根据有关法律和
中华人民共和国缔结或者参加的国际条约、协定，或者按照
平等互惠原则，处理外国司法或者执法机构关于提供存储于

境内个人信息的请求。非经中华人民共和国主管机关批准，
个人信息处理者不得向外国司法或者执法机构提供存储于
中华人民共和国境内的个人信息。 
第四十二条  境外的组织、个人从事侵害中华人民共和
国公民的个人信息权益，或者危害中华人民共和国国家安全、
公共利益的个人信息处理活动的，国家网信部门可以将其列
入限制或者禁止个人信息提供清单，予以公告，并采取限制
或者禁止向其提供个人信息等措施。 
第四十三条  任何国家或者地区在个人信息保护方面对
中华人民共和国采取歧视性的禁止、限制或者其他类似措施
的，中华人民共和国可以根据实际情况对该国家或者地区对
等采取措施。 
第四章 个人在个人信息处理活动中的权利 
第四十四条  个人对其个人信息的处理享有知情权、决
定权， 有权限制或者拒绝他人对其个人信息进行处理； 法律、
行政法规另有规定的除外。 
第四十五条  个人有权向个人信息处理者查阅、复制其
个人信息；有本法第十八条第一款、第三十五条规定情形的
除外。 
个人请求查阅、复制其个人信息的，个人信息处理者应
当及时提供。 
个人请求将个人信息转移至其指定的个人信息处理者，

符合国家网信部门规定条件的，个人信息处理者应当提供转
移的途径。 
第四十六条  个人发现其个人信息不准确或者不完整的，
有权请求个人信息处理者更正、补充。 
个人请求更正、补充其个人信息的，个人信息处理者应
当对其个人信息予以核实，并及时更正、补充。 
第四十七条  有下列情形之一的，个人信息处理者应当
主动删除个人信息；个人信息处理者未删除的，个人有权请
求删除： 
（一）处理目的已实现、无法实现或者为实现处理目的
不再必要； 
（二）个人信息处理者停止提供产品或者服务，或者保
存期限已届满； 
（三）个人撤回同意； 
（四）个人信息处理者违反法律、行政法规或者违反约
定处理个人信息； 
（五）法律、行政法规规定的其他情形。 
法律、行政法规规定的保存期限未届满，或者删除个人
信息从技术上难以实现的，个人信息处理者应当停止除存储
和采取必要的安全保护措施之外的处理。 
第四十八条  个人有权要求个人信息处理者对其个人信
息处理规则进行解释说明。 
第四十九条 自然人死亡的，其近亲属为了自身的合法、

正当利益，可以对死者的相关个人信息行使本章规定的查阅、
复制、更正、删除等权利；死者生前另有安排的除外。 
第五十条  个人信息处理者应当建立便捷的个人行使权
利的申请受理和处理机制。拒绝个人行使权利的请求的，应
当说明理由。 
个人信息处理者拒绝个人行使权利的请求的，个人可以
依法向人民法院提起诉讼。 
第五章 个人信息处理者的义务 
第五十一条  个人信息处理者应当根据个人信息的处理
目的、处理方式、个人信息的种类以及对个人权益的影响、
可能存在的安全风险等，采取下列措施确保个人信息处理活
动符合法律、行政法规的规定，并防止未经授权的访问以及
个人信息泄露、篡改、丢失： 
（一）制定内部管理制度和操作规程； 
（二）对个人信息实行分类管理； 
（三）采取相应的加密、去标识化等安全技术措施； 
（四）合理确定个人信息处理的操作权限，并定期对从
业人员进行安全教育和培训； 
（五）制定并组织实施个人信息安全事件应急预案； 
（六）法律、行政法规规定的其他措施。 
第五十二条  处理个人信息达到国家网信部门规定数量
的个人信息处理者应当指定个人信息保护负责人，负责对个

人信息处理活动以及采取的保护措施等进行监督。 
个人信息处理者应当公开个人信息保护负责人的联系
方式，并将个人信息保护负责人的姓名、联系方式等报送履
行个人信息保护职责的部门。 
第五十三条  本法第三条第二款规定的中华人民共和国
境外的个人信息处理者，应当在中华人民共和国境内设立专
门机构或者指定代表，负责处理个人信息保护相关事务，并
将有关机构的名称或者代表的姓名、联系方式等报送履行个
人信息保护职责的部门。 
第五十四条  个人信息处理者应当定期对其处理个人信
息遵守法律、行政法规的情况进行合规审计。 
第五十五条  有下列情形之一的，个人信息处理者应当
事前进行个人信息保护影响评估，并对处理情况进行记录： 
（一）处理敏感个人信息； 
（二）利用个人信息进行自动化决策； 
（三）委托处理个人信息、向其他个人信息处理者提供
个人信息、公开个人信息； 
（四）向境外提供个人信息； 
（五）其他对个人权益有重大影响的个人信息处理活动。 
第五十六条  个人信息保护影响评估应当包括下列内容： 
（一）个人信息的处理目的、处理方式等是否合法、正
当、必要； 
（二）对个人权益的影响及安全风险；

（三）所采取的保护措施是否合法、有效并与风险程度
相适应。 
个人信息保护影响评估报告和处理情况记录应当至少
保存三年。 
第五十七条 发生或者可能发生个人信息泄露、 篡改、 丢
失的，个人信息处理者应当立即采取补救措施，并通知履行
个人信息保护职责的部门和个人。通知应当包括下列事项： 
（一）发生或者可能发生个人信息泄露、篡改、丢失的
信息种类、原因和可能造成的危害； 
（二）个人信息处理者采取的补救措施和个人可以采取
的减轻危害的措施； 
（三）个人信息处理者的联系方式。 
个人信息处理者采取措施能够有效避免信息泄露、篡改、
丢失造成危害的，个人信息处理者可以不通知个人；履行个
人信息保护职责的部门认为可能造成危害的，有权要求个人
信息处理者通知个人。 
第五十八条 提供重要互联网平台服务、用户数量巨大、
业务类型复杂的个人信息处理者，应当履行下列义务： 
（一）按照国家规定建立健全个人信息保护合规制度体
系，成立主要由外部成员组成的独立机构对个人信息保护情
况进行监督； 
（二）遵循公开、公平、公正的原则，制定平台规则，
明确平台内产品或者服务提供者处理个人信息的规范和保

护个人信息的义务； 
（三）对严重违反法律、行政法规处理个人信息的平台
内的产品或者服务提供者，停止提供服务； 
（四）定期发布个人信息保护社会责任报告，接受社会
监督。 
第五十九条  接受委托处理个人信息的受托人，应 当依
照本法和有关法律、行政法规的规定，采取必要措施保障所
处理的个人信息的安全，并协助个人信息处理者履行本法规
定的义务。 
第六章 履行个人信息保护职责的部门 
第六十条  国家网信部门负责统筹协调个人信息保护工
作和相关监督管理工作。国务院有关部门依照本法和有关法
律、行政法规的规定，在各自职责范围内负责个人信息保护
和监督管理工作。 
县级以上地方人民政府有关部门的个人信息保护和监
督管理职责，按照国家有关规定确定。 
前两款规定的部门统称为履行个人信息保护职责的部
门。 
第六十一条  履行个人信息保护职责的部门履行下列个
人信息保护职责： 
（一）开展个人信息保护宣传教育，指导、监督个人信
息处理者开展个人信息保护工作；

（二）接受、处理与个人信息保护有关的投诉、举报； 
（三）组织对应用程序等个人信息保护情况进行测评，
并公布测评结果； 
（四）调查、处理违法个人信息处理活动； 
（五）法律、行政法规规定的其他职责。 
第六十二条  国家网信部门统筹协调有关部门依据本法
推进下列个人信息保护工作： 
（一）制定个人信息保护具体规则、标准； 
（二）针对小型个人信息处理者、处理敏感个人信息以
及人脸识别、人工智能等新技术、新应用，制定专门的个人
信息保护规则、标准； 
（三）支持研究开发和推广应用安全、方便的电子身份
认证技术，推进网络身份认证公共服务建设； 
（四）推进个人信息保护社会化服务体系建设，支持有
关机构开展个人信息保护评估、认证服务； 
（五）完善个人信息保护投诉、举报工作机制。 
第六十三条  履行个人信息保护职责的部门履行个人信
息保护职责，可以采取下列措施： 
（一）询问有关当事人，调查与个人信息处理活动有关
的情况; 
（二）查阅、复制当事人与个人信息处理活动有关的合
同、记录、账簿以及其他有关资料; 
（三）实施现场检查，对涉嫌违法的个人信息处理活动

进行调查; 
（四） 检查与个人信息处理活动有关的设备、 物品;对有
证据证明是用于违法个人信息处理活动的设备、物品，向本
部门主要负责人书面报告并经批准，可以查封或者扣押。 
履行个人信息保护职责的部门依法履行职责，当事人应
当予以协助、配合，不得拒绝、阻挠。 
第六十四条  履行个人信息保护职责的部门在履行职责
中，发现个人信息处理活动存在较大风险或者发生个人信息
安全事件的，可以按照规定的权限和程序对该个人信息处理
者的法定代表人或者主要负责人进行约谈，或者要求个人信
息处理者委 托专业机构对其个人信息处理活动进行合规审
计。个人信息处理者应当按照要求采取措施，进行整改，消
除隐患。 
履行个人信息保护职责的部门在履行职责中，发现违法
处理个人信息涉嫌犯罪的，应当及时移送公安机关依法处理。 
第六十五条  任何组织、个人有权对违法个人信息处理
活动向履行个人信息保护职责的部门进行投诉、举报。收到
投诉、举报的部门应当依法及时处理，并将处理结果告知投
诉、举报人。 
履行个人信息保护职责的部门应当公布接受投诉、举报
的联系方式。

第七章 法律责任 
第六十六条  违反本法规定处理个人信息，或者处理个
人信息未履行本法规定的个人信息保护义务的，由履行个人
信息保护职责的部门责令改正，给予警告，没收违法所得，
对违法处理个人信息的应用程序，责令暂停或者终止提供服
务；拒不改正的，并处一百万元以下罚款；对直接负责的主
管人员和其他直接责任人员处一万元以上十万元以下罚款。 
有前款规定的违法行为，情节严重的，由省级以上履行
个人信息保护职责的部门责令改正，没收违法所得，并处五
千万元以下或者上一年度营业额百分之五以下罚款，并可以
责令暂停相关业务或者停业整顿、通报有关主管部门吊销相
关业务许可或者吊销营业执照；对直接负责的主管人员和其
他直接责任人员处十万元以上一百万元以下罚款，并可以决
定禁止其在一定期限内担任相关企业的董事、监事、高级管
理人员和个人信息保护负责人。 
第六十七条 有本法规定的违法行为的，依照有关法律、
行政法规的规定记入信用档案，并予以公示。 
第六十八条  国家机关不履行本法规定的个人信息保护
义务的，由其上级机关或者履行个人信息保护职责的部门责
令改正；对直接负责的主管人员和其他直接责任人员依法给
予处分。 
履行个人信息保护职责的部门的工作人员玩忽职守、滥
用职权、徇私舞弊，尚不构成犯罪的，依法给予处分。

第六十九条  处理个人信息侵害个人信息权益造成损害，
个人信息处理者不能证明自己没有过错的，应当承担损害赔
偿等侵权责任。 
前款规定的损害赔偿责任按照个人因此受到的损失或
者个人信息处理者因此获得的利益确定 ;个人因此受到的损
失和个人信息处理者因此获得的利益难以确定的，根据实际
情况确定赔偿数额。 
第七十条  个人信息处理者违反本法规定处理个人信息，
侵害众多个人的权益的，人民检察院、法律规定的消费者组
织和由国家网信部门确定的组织可以依法向人民法院提起
诉讼。 
第七十一条 违反本法规定，构成违反治安管理行为的，
依法给予治安管理处罚；构成犯罪的， 依法追究刑事责任。 
第八章 附 则 
第七十二条  自然人因个人或者家庭事务处理个人信息
的，不适用本法。 
法律对各级人民政府及其有关部门组织实施的统计、档
案管理活动中的个人信息处理有规定的，适用其规定。 
第七十三条 本法下列用语的含义： 
（一）个人信息处理者，是指在个人信息处理活动中自
主决定处理目的、处理方式的组织、个人。 
（二）自动化决策，是指通过计算机程序自动分析、评

估个人的行为习惯、 兴趣爱好或者经济、 健康、 信用状况等，
并进行决策的活动。 
（三）去标识化，是指个人信息经过处理，使其在不借
助额外信息的情况下无法识别特定自然人的过程。 
（四）匿名化，是指个人信息经过处理无法识别特定自
然人且不能复原的过程。 
第七十四条 本法自 2021 年 11 月 1 日起施行。
