from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel


# 聊天机器人案例
# 创建模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=200)


# 生成一些结构化的数据： 5个步骤
# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int  # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、 提供一些样例数据，给AI
examples = [
    {
        "example": "Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
    },
    {
        "example": "Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
    },
    {
        "example": "Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"
    },
]

# 定义更明确的提示模板
template = """你是一个医疗数据生成助手，请生成符合以下要求的 {subject} 数据：
{extra}

要求：
1. 使用严格JSON格式输出
2. 键名对应字段：
- patient_id (数字)
- patient_name (中文名)
- diagnosis_code (疾病代码)
- procedure_code (操作代码) 
- total_charge (数字，无符号)
- insurance_claim_amount (数字，无符号)

示例：
{examples}

请生成 {runs} 条数据："""

prompt = ChatPromptTemplate.from_template(template)

# 创建生成链
chain = prompt | model | JsonOutputParser()

# 调用生成
result = chain.invoke({
    "subject": "医疗账单",
    "extra": "医疗总费用呈现正态分布，最小的总费用为1000",
    "examples": "\n".join([ex["example"] for ex in examples]), # 将 examples 列表中的每个示例内容提取出来，并用换行符 \n 连接成一个多行字符串。
    "runs": 10
})

for item in result:
    print(item)