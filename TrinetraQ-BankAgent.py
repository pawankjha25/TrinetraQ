import numpy as np
from utils import convert_rows_to_transaction_details
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain_core.agents import AgentAction
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

# Import your custom layers
from models.trinetraModel import MultiHeadSelfAttention, TrinetraTransformer

# Use custom objects to load the model
with CustomObjectScope({'MultiHeadSelfAttention': MultiHeadSelfAttention, 'TrinetraTransformer': TrinetraTransformer}):
    trinetra_model = load_model('../trinetra_model.h5', compile=False)


# Tool 1: Risk Scoring Tool
def risk_scoring_tool(transaction_details: dict) -> str:
    """
    Calculate fraud/risk score using the TrinetraTransformer model.
    Args:
        transaction_details (dict): Dictionary containing transaction features.

    Returns:
        str: Risk score.
    """
    # Prepare the input data (reshape or preprocess as required by your model)
    input_data = np.array([list(transaction_details.values())])

    # Predict risk score
    risk_score = trinetra_model.predict(input_data)[0][0]
    return f"Risk Score: {risk_score:.2f}"


# Tool 2: Action Execution Tool
def action_execution_tool(risk_score: str) -> str:
    """
    Perform actions based on the risk score.
    Args:
        risk_score (str): Risk score from the scoring tool.

    Returns:
        str: Action taken based on the risk score.
    """
    score = float(risk_score.split(": ")[1])
    if score < 0.3:
        return "Transaction Approved."
    elif score < 0.7:
        return "Transaction flagged for manual review."
    else:
        return "Transaction Blocked or Denied."


# Define Tools properly with `name`, `func`, and `description`
tools = [
    Tool(
        name="risk_scoring",
        func=risk_scoring_tool,
        description="Calculates the risk score for a transaction based on its details."
    ),
    Tool(
        name="action_execution",
        func=action_execution_tool,
        description="Decides action based on the risk score (approve, flag, or block)."
    )
]


# Define a custom agent class that will use the transformer model for decision making
class CustomTransformerAgent(BaseSingleActionAgent):
    def plan(self, intermediate_steps, **kwargs):
        transaction_details = kwargs.get("transaction_details")
        if not transaction_details:
            raise ValueError("Transaction details are required.")

        # Use the TrinetraTransformer to process the transaction
        print("****************")
        input_data = np.array([list(transaction_details.values())])  # Prepare data for prediction
        risk_score = trinetra_model.predict(input_data)[0][0]  # Get the risk score
        # Generate the action based on the risk score
        if risk_score < 0.3:
            action = "Transaction Approved."
        elif risk_score < 0.7:
            action = "Transaction flagged for manual review."
        else:
            action = "Transaction Blocked or Denied."

        # Return a comprehensive decision
        return AgentAction(
            tool="action_execution",
            tool_input=f"Risk Score: {risk_score:.2f}",
            log=f"Risk Score: {risk_score:.2f}, Action: {action}"
        )

    def aplan(self, intermediate_steps, **kwargs):
        transaction_details = kwargs.get("transaction_details")
        if not transaction_details:
            raise ValueError("Transaction details are required.")

        # Use the TrinetraTransformer to process the transaction
        print("****************")
        input_data = np.array([list(transaction_details.values())])  # Prepare data for prediction
        risk_score = trinetra_model.predict(input_data)[0][0]  # Get the risk score

        # Generate the action based on the risk score
        if risk_score < 0.3:
            action = "Transaction Approved."
        elif risk_score < 0.7:
            action = "Transaction flagged for manual review."
        else:
            action = "Transaction Blocked or Denied."

        # Return a comprehensive decision
        return AgentAction(
            tool="action_execution",
            tool_input=f"Risk Score: {risk_score:.2f}",
            log=f"Risk Score: {risk_score:.2f}, Action: {action}"
        )

    @property
    def input_keys(self):
        return ["transaction_details"]


# Create the AgentExecutor with the tools and the custom agent
agent = CustomTransformerAgent()
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)


# transactions = [
#     {"Time": 1000, "Amount": 5000, "Type": "Submit Cheque", "Location": "Bank A"},
#     {"Time": 200, "Amount": 200, "Type": "Credit Card Payment", "Location": "Online Store X"},
#     {"Time": 300, "Amount": 300, "Type": "Debit Card Withdrawal", "Location": "ATM"},
#     {"Time": 400, "Amount": 50000, "Type": "Loan Application", "Location": "Bank B"},
# ]


rows = [
    [29, 1.11088034163339, 0.168716770722767, 0.517143960377807, 1.32540691997371, -0.191573353787583, 0.0195037226488424, -0.0318491084003128, 0.117619919555324, 0.017664720727696, 0.0448647914479061, 1.3450747987323, 1.28633962057665, -0.252267065685462, 0.274457682308765, -0.810394372378945, -0.587005063447401, 0.0874510738489092, -0.550473628153257, -0.1547493550961, -0.19011971084361, -0.0377086544989231, 0.0957014620432248, -0.0481976468634584, 0.232114939125133, 0.606200748965636, -0.342096828961882, 0.0367696053150443, 0.0074799607310723, 6.54],
    [32, 1.24905471963177, -0.624727077037783, -0.710588903536079, -0.991600360912692, 1.42997319213398, 3.69297701929891, -1.09020864122523, 0.967290815452715, 0.850148519454057, -0.307081111779614, -0.456245308321934, 0.229981349844581, -0.0169130995868782, -0.220846086114885, 0.362417698896289, 0.315222304210749, -0.512265445432752, 0.11899461869584, 0.574720159539024, 0.0978526921491637, -0.00629271415563999, 0.00920022451060978, -0.129463200605864, 1.11297017048455, 0.500382108623099, 1.19654919930281, -0.0482204354357297, 0.00509362198376849, 29.89],
    [406,-2.312226542,1.951992011,-1.609850732, 3.997905588 ,	-0.522187865 ,	-1.426545319 ,	-2.537387306 ,	1.391657248	, -2.770089277 ,	-2.772272145 ,	3.202033207	, -2.899907388 ,	-0.595221881 ,	-4.289253782 ,	0.38972412 ,	-1.14074718	, -2.830055675 ,	-0.016822468 ,	0.416955705 ,	0.126910559, 	0.517232371	, -0.035049369 ,	-0.465211076 ,	0.320198199	, 0.044519167 ,	0.177839798	, 0.261145003 ,	-0.143275875,	0	]
]

# Convert rows to transaction details
transactions = convert_rows_to_transaction_details(rows)
# Process Transactions
for transaction in transactions:
    print(f"\nProcessing Transaction: {transaction}")
    # Run the agent for each transaction
    response = agent_executor.run(transaction_details=transaction)
    print(f"Response: {response}")
