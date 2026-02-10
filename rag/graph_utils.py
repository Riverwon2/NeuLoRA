"""
LangGraph 유틸리티 함수

langchain_teddynote 의존성을 완전히 제거하기 위한 대체 구현:
  - random_uuid()       → langchain_teddynote.messages.random_uuid 대체
  - visualize_graph()   → langchain_teddynote.graphs.visualize_graph 대체
  - invoke_graph()      → langchain_teddynote.messages.invoke_graph 대체
  - stream_graph()      → langchain_teddynote.messages.stream_graph 대체
"""

import uuid
from typing import Any, Dict, Optional


def random_uuid() -> str:
    """랜덤 UUID 생성 (langchain_teddynote.messages.random_uuid 대체)"""
    return uuid.uuid4().hex


def visualize_graph(app):
    """
    LangGraph 앱의 그래프를 시각화합니다.
    (langchain_teddynote.graphs.visualize_graph 대체)

    Jupyter Notebook 환경에서 그래프를 이미지로 표시합니다.
    """
    from IPython.display import Image, display

    try:
        img_data = app.get_graph().draw_mermaid_png()
        display(Image(img_data))
    except Exception as e:
        print(f"⚠️ 그래프 시각화 실패: {e}")
        print("Tip: graphviz 또는 mermaid 관련 패키지가 필요할 수 있습니다.")
        # 대안: 텍스트 기반 그래프 출력
        print(app.get_graph().draw_ascii())


def invoke_graph(
    app,
    inputs: Dict[str, Any],
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    LangGraph 앱을 실행하고 결과를 출력합니다.
    (langchain_teddynote.messages.invoke_graph 대체)

    Args:
        app: 컴파일된 LangGraph 앱
        inputs: 입력 상태
        config: RunnableConfig
        verbose: 상세 출력 여부

    Returns:
        실행 결과 딕셔너리
    """
    # stream 모드로 실행하여 각 노드별 출력 표시
    for event in app.stream(inputs, config=config):
        for node_name, node_output in event.items():
            if verbose:
                print()
                print("=" * 50)
                print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                print("- " * 25)

                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        val_str = str(value)
                        if len(val_str) > 500:
                            val_str = val_str[:500] + "..."
                        print(f"\033[1;32m{key}\033[0m:\n {val_str}")
                else:
                    print(f"  {node_output}")

    # 최종 상태 반환
    final_state = app.get_state(config).values
    return final_state


def stream_graph(
    app,
    inputs: Dict[str, Any],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    LangGraph 앱을 스트리밍 실행하고 결과를 출력합니다.
    (langchain_teddynote.messages.stream_graph 대체)

    Args:
        app: 컴파일된 LangGraph 앱
        inputs: 입력 상태
        config: RunnableConfig

    Returns:
        최종 상태 딕셔너리
    """
    for event in app.stream(inputs, config=config):
        for node_name, node_output in event.items():
            print()
            print("=" * 50)
            print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
            print("- " * 25)

            if isinstance(node_output, dict):
                for key, value in node_output.items():
                    val_str = str(value)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    print(f"\033[1;32m{key}\033[0m:\n {val_str}")
            else:
                print(f"  {node_output}")

    # 최종 상태 반환
    final_state = app.get_state(config).values
    return final_state
