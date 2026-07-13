import torch
import torch.nn.functional as F


def predict_with_confidence_routing(model, input_tensor, class_names, confidence_threshold=0.6):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

        top_prob, top_idx = torch.max(probabilities, dim=0)
        top_prob = top_prob.item()
        top_class = class_names[top_idx.item()]

        top3_probs, top3_idxs = torch.topk(probabilities, 3)
        top3_results = [
            (class_names[idx.item()], prob.item())
            for prob, idx in zip(top3_probs, top3_idxs)
        ]

    needs_review = top_prob < confidence_threshold

    result = {
        "predicted_class": top_class,
        "confidence": top_prob,
        "needs_manual_review": needs_review,
        "top3_predictions": top3_results
    }

    return result


def print_prediction_result(result):
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")

    if result["needs_manual_review"]:
        print("⚠️  LOW CONFIDENCE — Flagged for manual review")
    else:
        print("✓ High confidence prediction")

    print("\nTop 3 predictions:")
    for cls, prob in result["top3_predictions"]:
        print(f"  {cls}: {prob:.1%}")