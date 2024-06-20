# Answer Extractor Module

::: pipelines.pipelines.nodes.answer_extractor.answer_extractor_preprocessor
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60

::: pipelines.pipelines.nodes.answer_extractor.answer_extractor
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60

::: pipelines.pipelines.nodes.answer_extractor.qa_filter_postprocessor
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60

::: pipelines.pipelines.nodes.answer_extractor.qa_filter
    options:
        summary: true
        separate_signature: true
        show_signature_annotations: true
        line_length: 60
