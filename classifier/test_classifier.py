from lca_classifier import LCAClassifier
import json

def main():
    print("Loading classifier with llama3.2 ...")
    clf = LCAClassifier(model="llama3.2")

    # Test 1 - Should be USEFUL
    print("\n----- Test 1: Theoretical Book -----")
    result1 = clf.classify(
        title="Life Cycle Assessment: Theory and Practice",
        content="This comprehensive book covers the theoretical foundations of LCA including goal and scope definition according to ISO 14040, life cycle inventory analysis, impact assessment methods, allocation procedures, uncertainty analysis and interpretation phase.",
        url="https://example.com/theoretical-book"
    )
    print(json.dumps(result1.__dict__, indent=2))

    # Test 2 - Should be NOT useful
    print("\n----- Test 2: Pure Case Study -----")
    result2 = clf.classify(
        title="LCA of a Plastic Bottle in India",
        content="We conducted a cradle-to-grave LCA of PET bottles produced in Maharashtra. Results show GWP of 1.2 kg CO2-eq per bottle. Data was collected from local factories.",
        url="https://example.com/case-study"
    )
    print(json.dumps(result2.__dict__, indent=2))

    print("\n----- Summary -----")
    print(f"Test 1 useful? → {result1.is_useful}  (expected: True)")
    print(f"Test 2 useful? → {result2.is_useful}  (expected: False)")

if __name__ == "__main__":
    main()