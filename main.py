
from Graph.workflow import build_graph


def main():
    graph = build_graph()

    user_query = input("You: ").strip()

    if not user_query:
        print("Please enter a task.")
        return

    result = graph.invoke({"user_query": user_query})

    print("\nAssistant:")
    print(result["final_answer"])
    
    

if __name__=="__main__":
    main()