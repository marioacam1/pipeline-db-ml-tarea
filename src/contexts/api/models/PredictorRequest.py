from pydantic import BaseModel, validator

class PredictorRequest(BaseModel):
    tipo_correo: str
    pais: str
    ciudad: str

    @validator("tipo_correo")
    def validate_tipo_correo(cls, v):
        if not v or not v.strip():
            raise ValueError("El tipo de correo no puede estar vacío.")
        return v.strip().lower()

    @validator("pais")
    def validate_pais(cls, v):
        if not v or not v.strip():
            raise ValueError("El país no puede estar vacío.")
        return v.strip()

    @validator("ciudad")
    def validate_ciudad(cls, v):
        if not v or not v.strip():
            raise ValueError("La ciudad no puede estar vacía.")
        return v.strip()

