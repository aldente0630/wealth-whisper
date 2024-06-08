import os
import sys
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from app.chatbot import ChatBot
from app.utils import get_name, load_config, logger


def get_dir_path(dir_name: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), dir_name))


if __name__ == "__main__":
    try:
        config_dir = get_dir_path(os.path.join(os.pardir, "app", "configs"))
        config = load_config(os.path.join(config_dir, "config.yaml"))

        # Create a ChatBot instance
        chatbot = ChatBot(
            profile_name=config.profile_name,
            region_name=config.region_name,
            domain_name=get_name(config.proj_name, config.stage, "docs"),
            index_name=get_name(config.proj_name, config.stage, "docs"),
            endpoint_name=get_name(config.proj_name, config.stage, "embedding"),
            secret_name=get_name(config.proj_name, config.stage, "docs"),
            embedding_model_name=config.embedding_model_name,
            retriever_type=config.retriever_type,
            k=config.k,
            multiplier=config.multiplier,
            semantic_weight=config.semantic_weight,
            use_hyde=config.use_hyde,
            use_rag_fusion=config.use_rag_fusion,
            use_parent_document=config.use_parent_document,
            use_time_weight=config.use_time_weight,
            decay_rate=config.decay_rate,
            use_reorder=config.use_reorder,
            temperature=config.temperature,
            verbose=True,
            logger=logger,
        )

        # Set up Streamlit layouts
        st.set_page_config(page_title="Wealth Whisper", page_icon="💰")
        st.title("💰 Wealth Whisper")
        st.subheader("Powered by AWS Professional Services Korea", divider="blue")
        st.caption(
            '애널리스트 리포트 정보를 검색하여 제공해 드립니다. 다음과 같은 질문에 답변할 수 있습니다. "삼성전자 예상 실적은?", "미국 금리 전망은?"'
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "안녕하세요, 저는 대한민국 금융 투자 정보를 제공하는 AI 어시스턴트 Wealth Whisper입니다. 무엇을 도와드릴까요?",
                }
            ]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Searching for information...")
        if query is not None:
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            # Set up a container to receive Bedrock streaming with a Streamlit callback handler
            st_callback = StreamlitCallbackHandler(
                st.container(), collapse_completed_thoughts=True
            )
            results = chatbot.invoke(query, st_callback=st_callback)

            # Save messages in a session and printing them out
            st.session_state.messages.append(
                {"role": "assistant", "content": results["answer"]}
            )
            st.chat_message("assistant").write(results["answer"])

            if results["source_descriptions"] is not None:
                source_descriptions = (
                    f"❓ 이 정보는 '{results['source_descriptions']}'에서 제공되었습니다."
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": source_descriptions}
                )
                st.chat_message("assistant").write(source_descriptions)

            st_callback._complete_current_thought()

    except Exception as error:
        st.error(f"The following error occurred: {error}")
