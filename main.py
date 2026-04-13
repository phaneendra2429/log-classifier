from src.inference import predict_log

def main():
    print("Log Classification System")
    print("Type 'exit' to stop\n")

    while True:
        log_input = input("Enter log message: ")

        if log_input.lower() == "exit":
            break

        result = predict_log(log_input)

        print("\nResult:")
        print("Log:", result["log"])
        print("Predicted Label:", result["predicted_label"])
        print("Confidence:", result["confidence"])
        print("Summary:", result["summary"])

        # show alternative prediction
        if "alternative" in result:
            print("Alternative:", result["alternative"])

        if "alternative_confidence" in result:
            print("Alt Confidence:", result["alternative_confidence"])

        print()

if __name__ == "__main__":
    main()