# Contrato de artefactos

Cada módulo dispone de una carpeta propia. Cuando tenga resultados reproducibles, debe añadir un
`manifest.json` que cumpla `docs/module_contract.schema.json`. Los datos tabulares y modelos
serializados se guardan en la misma carpeta del módulo o en la ubicación documentada por el manifest.

Streamlit no entrena modelos: consulta estos artefactos y los representa. No se deben versionar
datasets grandes ni información sensible.
