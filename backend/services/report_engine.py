def generate_verification_report(
    file_info: dict,
    extracted_fields: dict,
    validation_results: dict,
    processing_time: float
) -> dict:
    """
    Constructs the final structured JSON response.
    Conforms to the API spec:
    {
      "documents": [{
        "document_type": ...,
        "original_size": ...,
        "optimized_size": ...,
        "compression_ratio": ...,
        "extracted_text": ...,
        "fields": {...},
        "confidence": 0.93,
        "verification_status": "verified"
      }],
      "processing_time_seconds": 3.2
    }
    """
    # Pull raw_text before cleaning it from the dict
    raw_text = extracted_fields.pop("raw_text", "")

    # Flatten fields from {field: {value, confidence}} → {field: value}
    flat_fields = {
        k: v.get("value") if isinstance(v, dict) else v
        for k, v in extracted_fields.items()
    }

    orig_mb = file_info.get("original_size_mb", 0) or 0
    opt_mb  = file_info.get("optimized_size_mb", 0) or 0
    comp    = file_info.get("compression_pct", 0) or 0

    aggregate_conf = validation_results.get("aggregate_confidence", 0.0)

    # Canonical report format
    report = {
        "status": "success",
        # Wrapped format matching the API spec
        "documents": [
            {
                "document_type":    file_info.get("filename", "unknown"),
                "original_size":    f"{orig_mb} MB",
                "optimized_size":   f"{opt_mb} MB",
                "compression_ratio": f"{comp}%",
                "extracted_text":   raw_text[:500] if raw_text else "",
                "fields":           flat_fields,
                "confidence":       aggregate_conf,
                "verification_status": validation_results.get("overall_status", "failed"),
            }
        ],
        "processing_time_seconds": round(processing_time, 2),
        # Full detail for the UI dashboard
        "data": {
            "file_info":          file_info,
            "extracted_fields":   extracted_fields,
            "validation":         validation_results,
            "processing_time_sec": round(processing_time, 2),
        }
    }

    return report
