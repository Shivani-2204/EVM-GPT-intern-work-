import streamlit as st 
import pandas as pd 
import json
import re
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Excel data 
def load_data():
    df = pd.read_excel(r"C:\Users\KIIT\Desktop\tata\MBP Latest.xlsx")

    df["Suggested MRC(monthly Recurring charge)"] = (
        df["Suggested MRC(monthly Recurring charge)"]
        .astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0").astype(float)
    )
    df["Suggested One time charge"] = (
        df["Suggested One time charge"]
        .astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0").astype(float)
    )
# changing and processing the bandwidth data to normalise using regex 
    def extract_bandwidth(value):
        if pd.isna(value):
            return None
        value_str = str(value).lower().strip()
        if "m" in value_str and "/" not in value_str:
            match = re.match(r"(\d+)", value_str)
            return int(match.group(1)) if match else None
        if "/" in value_str:
            first_part = value_str.split("/")[0]
            match = re.match(r"(\d+)", first_part)
            return int(match.group(1)) if match else None
        match = re.match(r"(\d+)", value_str)
        return int(match.group(1)) if match else None
#cleaning data 
    df["IP Bandwidth quoted (mbps)"] = df["IP Bandwidth quoted (mbps)"].apply(extract_bandwidth)

    df["City_clean"] = df["City"].astype(str).str.lower().str.strip()
    df["Region_clean"] = df["Region"].astype(str).str.lower().str.strip()
    df["Country_clean"] = df["Country"].astype(str).str.lower().str.strip()
    df["Flavour_clean"] = df["*evm IW flavour"].astype(str).str.lower().str.replace("-", "").str.strip()
    df["Flavour Type_clean"] = df["Flavour Type"].astype(str).str.lower().str.strip()
    df["Data_Allowance_clean"] = df["Data Allowance"].astype(str).str.lower().str.strip()

    return df

# call api 
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

#  user query
def ask_qwen(prompt):
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the following fields from the user query and return them in JSON format. "
                        "Always include all fields in the output. If a field is not present in the query, return null for that field.\n\n"
                        "Required JSON structure:\n"
                        "{\n"
                        "  \"city\": <string or null>,\n"
                        "  \"region\": <string or null>,\n"
                        "  \"country\": <string or null>,\n"
                        "  \"bandwidth\": <number or null>,\n"
                        "  \"*evm IW flavour\": <string or null>,\n"
                        "  \"Flavour Type\": <string or null>,\n"
                        "  \"data_allowance\": <string or null>,\n"
                        "  \"budget\": <number or null>\n"
                        "}\n\n"
                        "Allowed values for *evm IW flavour:\n"
                        "- basic, standard, connectdia, enhanced, connectbroadband, wireless, satellite\n\n"
                        "Allowed values for Flavour Type:\n"
                        "- broadband, dia, satellite, wireless\n\n"
                        "Allowed values for data_allowance:\n"
                        "- Unlimited\n"
                        "- 500 GB (FUP of 1 Mbps after data exhaust)\n\n"
                        "Rules:\n"
                        "- Match keywords exactly or through context (e.g., 'enhanced link' → 'enhanced')\n"
                        "- If a field is not clearly stated, set it to null\n"
                        "- Never make assumptions or infer fields\n"
                        "- If the query mentions a Flavour Type like 'DIA', 'broadband', 'satellite', or 'wireless', but does not clearly mention a specific '*evm IW flavour' like 'connectdia', 'connectbroadband', etc., then:\n"
                        "    - Set 'Flavour Type' accordingly (e.g., 'dia')\n"
                        "    - Set '*evm IW flavour' to null\n"
                        "- If the user uses words not in the allowed lists, ignore them (set to null)\n"
                        "- If the user query asks for bandwidth in 'G' (e.g., 1G, 2G), interpret it as 1000, 2000 Mbps respectively"
                        
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Summary generation using Qwen
def summarize_with_qwen(user_query, filtered_data):
    try:
        summary_prompt = (
            "You are a summarizer for internet plan queries. Based on the user's request and the list of matching filtered plans, "
            "generate a clear and informative summary. The summary should include:\n"
            "1. The query asked by the user.\n"
            "2. The best price option.\n"
            "3. All unique *evm IW flavour options and their prices from the filtered data.\n"
            "Only summarize what's asked. Do not assume or infer any data.\n"
            "Always include this note at the end: 'Note: The prices are for referential purposes and can vary based on vendors/partners.'\n\n"
            f"User query: {user_query}\n"
            f"Filtered plans: {json.dumps(filtered_data, indent=2)}"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[{"role": "system", "content": summary_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

# RAG setup with LangChain
@st.cache_resource
def build_vectorstore(df):
    documents = []
    for _, row in df.iterrows():
        content = f"""
        City: {row['City']}
        Region: {row['Region']}
        Country: {row['Country']}
        Bandwidth: {row['IP Bandwidth quoted (mbps)']}
        Flavour: {row['*evm IW flavour']}
        Flavour Type: {row['Flavour Type']}
        MRC: {row['Suggested MRC(monthly Recurring charge)']}
        OTC: {row['Suggested One time charge']}
        Data Allowance: {row.get('Data Allowance', '')}
        """
        documents.append(Document(page_content=content))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def rag_answer(user_query, vectorstore):
    try:
        docs = vectorstore.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "You are a helpful assistant recommending internet plans. "
            "Use the context below to answer the user's question precisely.\n\n"
            f"Context:\n{context}\n\n"
            f"User question: {user_query}\n"
            "Answer:"
        )

        response = client.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"RAG error: {e}"

# Streamlit UI
st.title("\U0001F310 EVM Chatbot – Internet Plan Recommender")

if st.button("\U0001F504 Reload Excel Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

df = load_data()
vectorstore = build_vectorstore(df)

query = st.text_input("Enter your query (e.g., 'Suggest a 200 Mbps plan in Paris')")

if st.button("Get Plan"):
    with st.spinner("Asking Qwen AI..."):
        qwen_output = ask_qwen(query)
        st.markdown("### Qwen's Extracted Fields")
        st.code(qwen_output)

        try:
            qwen_dict = json.loads(qwen_output)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", qwen_output)
            if match:
                qwen_dict = json.loads(match.group())
            else:
                st.error("Qwen response could not be parsed.")
                st.stop()

        city = (qwen_dict.get("city") or "").strip().lower()
        region = (qwen_dict.get("region") or "").strip().lower()
        country = (qwen_dict.get("country") or "").strip().lower()
        bandwidth = int(qwen_dict.get("bandwidth") or 0)
        budget = float(qwen_dict.get("budget") or 0)
        flavour = re.sub(r"[^a-z0-9]", "", (qwen_dict.get("*evm IW flavour") or "").lower().strip())
        flavour_type = (qwen_dict.get("Flavour Type") or "").strip().lower()
        data_allowance = (qwen_dict.get("data_allowance") or "").strip().lower()
        
# add filters 
        filtered_df = df.copy()
        if city:
            filtered_df = filtered_df[filtered_df["City_clean"] == city]
        if region:
            filtered_df = filtered_df[filtered_df["Region_clean"] == region]
        if country:
            filtered_df = filtered_df[filtered_df["Country_clean"] == country]
        if bandwidth:
            filtered_df = filtered_df[filtered_df["IP Bandwidth quoted (mbps)"] == bandwidth]
        if budget:
            filtered_df = filtered_df[filtered_df["Suggested MRC(monthly Recurring charge)"] <= budget]
        if flavour:
            filtered_df = filtered_df[filtered_df["Flavour_clean"].str.contains(flavour, na=False)]
        if flavour_type:
            filtered_df = filtered_df[filtered_df["Flavour Type_clean"] == flavour_type]
        if data_allowance:
            filtered_df = filtered_df[filtered_df["Data_Allowance_clean"] == data_allowance]

        filtered_df = filtered_df.dropna(subset=[
            "Suggested MRC(monthly Recurring charge)",
            "Suggested One time charge",
            "*evm IW flavour"
        ])

        st.markdown("### Filtered Data")
        st.dataframe(filtered_df)

        if not filtered_df.empty:
            top_rows = filtered_df.nsmallest(3, "Suggested MRC(monthly Recurring charge)")
            recommendations = []
            for _, row in top_rows.iterrows():
                recommendations.append({
                    "City": row["City"],
                    "Region": row["Region"],
                    "Country": row["Country"],
                    "Bandwidth": row["IP Bandwidth quoted (mbps)"],
                    "Flavour": row["*evm IW flavour"],
                    "Flavour Type": row["Flavour Type"],
                    "MRC": row["Suggested MRC(monthly Recurring charge)"],
                    "OTC": row["Suggested One time charge"],
                    "Data Allowance": row.get("Data Allowance")
                })

            st.markdown("### \U0001F4A1 Recommended Plans")
            st.json(recommendations)

            st.markdown("### \U0001F50A Summary")
            summary = summarize_with_qwen(query, recommendations)
            st.write(summary)

        else:
            st.warning("No matching plans found.")

        st.markdown("### \U0001F9E0 RAG-Based Semantic Answer")
        rag_resp = rag_answer(query, vectorstore)
        st.write(rag_resp)
