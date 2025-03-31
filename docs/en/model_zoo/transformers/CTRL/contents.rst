CTRL Model Summary  
------------------------------------  

The following table summarizes the CTRL models and corresponding pretrained weights currently supported by PaddleNLP.  

+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+  
| Pretrained Weight                                                                | Language     | Details of the model                                                             |  
+==================================================================================+==============+==================================================================================+  
|``ctrl``                                                                          | English      | 48-layer, 1280-hidden,                                                           |  
|                                                                                  |              | 16-heads, 1701M parameters.                                                      |  
|                                                                                  |              | The CTRL base model.                                                             |  
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+  
|``sshleifer-tiny-ctrl``
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+
| You are a professional NLP technical translator. Translate Chinese to English while:                                                                 |
| 1. Preserving EXACT formatting (markdown/rst/code)                                                                                     |
| 2. Keeping technical terms in English                                                                                                  |
| 3. Maintaining code/math blocks unchanged                                                                                              |
| 4. Using proper academic grammar                                                                                                       |
| 5. Keep code blocks in documents original                                                                                              |
| 6. Keep the link in markdown/rst the same. E.g. [链接](#这里) translates to [link](#这里) not [link](#here)                             |
| 7. Keep the html tag in markdown/rst the same.                                                                                          |
| 8. Return only the translation result, no additional messages.                                                                         |
+----------------------------------------------------------------------------------+--------------+----------------------------------------------------------------------------------+ 

| English      | 2-layer, 16-hidden,                                                              |
|              | 2-heads, 5M parameters.                                                          |
|              | The Tiny CTRL model.                                                             |
+----------------------------------------------------------------------------------+