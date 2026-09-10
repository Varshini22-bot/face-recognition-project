"""Public API response models."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
	code: str
	message: str


class ErrorResponse(BaseModel):
	success: bool = False
	error: ErrorDetail


class FaceResponse(BaseModel):
	recognized: bool
	name: str | None
	similarity: float | None
	threshold: float
	status: str


class RecognitionResponse(BaseModel):
	success: bool = True
	faces: list[FaceResponse]
	face_count: int
	processing_time_ms: float


class PersonResponse(BaseModel):
	id: int
	name: str
	created_at: str


class PeopleResponse(BaseModel):
	people: list[PersonResponse]
	count: int


class RegistrationResponse(BaseModel):
	success: bool = True
	person: PersonResponse


class DeleteResponse(BaseModel):
	success: bool = True
	deleted_id: int