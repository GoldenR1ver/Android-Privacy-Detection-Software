

---

# 第一层：法律知识库（Legal Knowledge Base）

这一层就是你已经整理好的表。

建议转成机器可读。

目录：

```text
knowledge_base/
├── categories.json
├── category_alias.json
├── sensitivity.json
└── laws.json
```

---

## categories.json

定义类别。

示例：

```json
{
  "identity": {
    "name": "身份与标识",
    "examples": [
      "姓名",
      "身份证号",
      "手机号",
      "邮箱"
    ]
  },

  "location": {
    "name": "位置与轨迹",
    "examples": [
      "GPS",
      "轨迹",
      "地址"
    ]
  },

  "biometric": {
    "name": "生物识别",
    "examples": [
      "人脸",
      "指纹",
      "声纹"
    ]
  }
}
```

---

## sensitivity.json

记录风险等级。

```json
{
    "identity":"MEDIUM",
    "location":"HIGH",
    "biometric":"HIGH",
    "behavior":"MEDIUM",
    "device":"LOW"
}
```

---

## category_alias.json

解决命名问题。

```json
{
 "imei":"identity",
 "android_id":"identity",

 "gps":"location",
 "lat":"location",
 "longitude":"location",

 "face":"biometric",

 "cookie":"online_identifier"
}
```

---

到这里法律知识库完成。

---

# 第二层：Android 检测规则映射表（核心）

新增：

```text
mapping/
└── android_rules.json
```

这是实验真正工作的地方。

结构：

```json
[
{
"signal":"permission",

"match":[
"android.permission.ACCESS_FINE_LOCATION"
],

"collects":[
"location"
],

"confidence":0.95
},

{
"signal":"api",

"match":[
"TelephonyManager.getDeviceId"
],

"collects":[
"identity"
],

"confidence":0.9
},

{
"signal":"sdk",

"match":[
"com.google.firebase.analytics"
],

"collects":[
"behavior"
],

"confidence":0.7
}
]
```

字段解释：

| 字段         | 含义       |
| ---------- | -------- |
| signal     | 证据类型     |
| match      | 匹配条件     |
| collects   | 对应个人信息类别 |
| confidence | 可信度      |

---

建议至少覆盖这几类：

---

## Permission → PI

```text
ACCESS_FINE_LOCATION
→ location

READ_CONTACTS
→ contact

READ_PHONE_STATE
→ identity

CAMERA
→ biometric

RECORD_AUDIO
→ biometric

READ_CALENDAR
→ schedule

READ_SMS
→ communication

BODY_SENSORS
→ health
```

---

## API → PI

```text
LocationManager
→ location

TelephonyManager
→ identity

AccountManager
→ account

ContactsContract
→ contact

CameraX
→ biometric

BluetoothAdapter
→ device

MediaRecorder
→ sensory

AdvertisingIdClient
→ online_identifier
```

---

## SDK → PI

```text
Firebase Analytics
→ behavior

AppsFlyer
→ advertising

Facebook SDK
→ profile

Bugly
→ device

Umeng
→ profile
```

---

## Network → PI

正则匹配：

```text
imei
deviceid
oaid
lat
lng
userid
cookie
token
email
phone
```

映射：

```json
{
"imei":"identity",
"lat":"location",
"cookie":"online_identifier"
}
```

---

# 以下是一个示例

可以，`laws.json` 我也建议删掉。

你现在不是做法律展示系统，而是做 **Lab-4-1 检测系统**。法律来源已经固化在知识里，不需要单独维护一个 `laws.json`（后面更新和维护成本高，而且检测阶段不会用到）。

建议最终知识库结构：

```text
knowledge_base/
├── categories.json
├── aliases.json
├── sensitivity.json
├── android_mapping.json
├── risk_rules.json
└── README.md
```

其中：

* `categories.json`
  —— 法律知识库（核心）
* `aliases.json`
  —— 同义词归一化
* `sensitivity.json`
  —— 风险等级
* `android_mapping.json`
  —— Android → PI 映射
* `risk_rules.json`
  —— 风险判定

---

## 1. categories.json（核心法律知识库）

建议结构：

```json
{
  "identity": {
    "display_name": "身份与标识",
    "sensitive": false,

    "subtypes": [

      {
        "name":"name",

        "examples":[
          "真实姓名",
          "曾用名",
          "昵称",
          "用户名"
        ],

        "keywords":[
          "name",
          "username",
          "nickname",
          "realname"
        ],

        "android_signals":[
          "AccountManager",
          "READ_PROFILE"
        ]
      },

      {
        "name":"identifier",

        "examples":[
          "身份证号",
          "护照号",
          "社保号",
          "驾照号"
        ],

        "keywords":[
          "id_number",
          "passport",
          "ssn",
          "license"
        ],

        "android_signals":[
          "TelephonyManager",
          "getDeviceId"
        ]
      },

      {
        "name":"contact",

        "examples":[
          "手机号",
          "邮箱",
          "住址"
        ],

        "keywords":[
          "phone",
          "email",
          "address"
        ],

        "android_signals":[
          "READ_CONTACTS",
          "ContactsContract"
        ]
      }
    ]
  },

  "location": {

    "display_name":"位置与轨迹",

    "sensitive":true,

    "subtypes":[

      {
        "name":"precise_location",

        "examples":[
          "GPS",
          "WiFi定位",
          "实时位置"
        ],

        "keywords":[
          "lat",
          "lng",
          "gps",
          "location"
        ],

        "android_signals":[
          "ACCESS_FINE_LOCATION",
          "LocationManager"
        ]
      },

      {
        "name":"trajectory",

        "examples":[
          "运动轨迹",
          "出行记录"
        ],

        "keywords":[
          "route",
          "path"
        ],

        "android_signals":[
          "FusedLocationProvider"
        ]
      }
    ]
  },

  "biometric": {

    "display_name":"生物识别",

    "sensitive":true,

    "subtypes":[

      {
        "name":"face",

        "examples":[
          "人脸",
          "面部特征"
        ],

        "keywords":[
          "face",
          "facial"
        ],

        "android_signals":[
          "CameraX",
          "MLKit"
        ]
      },

      {
        "name":"voice",

        "examples":[
          "声纹"
        ],

        "keywords":[
          "voiceprint"
        ],

        "android_signals":[
          "MediaRecorder"
        ]
      }
    ]
  }
}
```

这里已经把：

```text
法律分类
↓
示例
↓
检测关键词
↓
Android证据
```

放进同一个对象。

---

## 2. aliases.json（统一命名）

这个比法律表重要。

```json
{
"imei":"identifier",
"meid":"identifier",

"email":"contact",
"mail":"contact",

"lat":"precise_location",
"lng":"precise_location",

"faceid":"face",

"cookie":"online_identifier",

"android_id":"online_identifier",

"oaid":"online_identifier",

"adid":"online_identifier",

"heart_rate":"health"
}
```

目的：

统一：

```text
DeviceID
IMEI
ANDROID_ID
OAID
↓
online_identifier
```

---

## 3. sensitivity.json

不要复杂。

```json
{
"identity":"MEDIUM",

"contact":"MEDIUM",

"location":"HIGH",

"biometric":"HIGH",

"health":"HIGH",

"finance":"HIGH",

"behavior":"MEDIUM",

"profile":"HIGH",

"device":"LOW"
}
```

---

## 4. android_mapping.json（重点）

这是下一阶段真正用的。

结构：

```json
[
{
"type":"permission",

"match":[
"android.permission.ACCESS_FINE_LOCATION"
],

"collects":[
"precise_location"
],

"confidence":0.95
},

{
"type":"api",

"match":[
"LocationManager.getLastKnownLocation"
],

"collects":[
"trajectory"
],

"confidence":0.9
},

{
"type":"api",

"match":[
"TelephonyManager.getDeviceId"
],

"collects":[
"identifier"
],

"confidence":0.95
},

{
"type":"sdk",

"match":[
"com.appsflyer"
],

"collects":[
"profile"
],

"confidence":0.7
}
]
```

---

## 5. risk_rules.json

最后做输出。

```json
{
"HIGH":[
"biometric",
"health",
"location",
"finance"
],

"MEDIUM":[
"identity",
"profile",
"behavior"
],

"LOW":[
"device"
]
}
```

---

最后实验流程：

```text
APK

↓

Exodus
权限/Tracker

↓

Androguard
API调用

↓

Frida
运行时

↓

mitmproxy
网络包

↓

Normalizer
统一字段

↓

Knowledge Base
(categories+aliases)

↓

Android Mapping

↓

Risk Rules

↓

Report
```
