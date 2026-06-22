# SpatialScope — единый отчёт по актуальным UML-диаграммам

## Назначение документа

Этот документ содержит актуальный комплект UML-диаграмм проекта `SpatialScope` в едином месте в виде исходного `PlantUML`-кода. Документ предназначен для:

- передачи другому AI-агенту;
- вставки фрагментов в отчёт;
- централизованного хранения исходников диаграмм;
- проверки согласованности между диаграммами.

В документ включены только актуальные диаграммы, соответствующие текущему набору файлов в `diploma/docs/uml` и `diploma/docs/uml/state`.

---

## Состав актуального UML-комплекта

### Структурные диаграммы
- диаграмма классов;
- диаграмма компонентов;
- диаграмма развёртывания.

### Поведенческие диаграммы
- диаграмма деятельности;
- набор диаграмм последовательностей;
- набор диаграмм вариантов использования;
- набор диаграмм состояний.

---

# 1. Диаграмма классов

## Файл
`diploma/docs/uml/class-diagram.puml`

```plantuml
@startuml
skinparam backgroundColor transparent
skinparam classAttributeIconSize 0
hide empty members

package "Persistence Layer" {
  class User
  class Project
  class XeniumSlide
  class MLModel
  class ModelPreset
  class AnalysisReport
  class AuditLog
  class DatabaseService
}

package "Registry & Orchestration" {
  abstract class AbstractModelRegistry
  class PresetConfiguration
  class ModelPresetRegistry

  class BaseSpatialOrchestrator
  class XeniumSystemOrchestrator
}

package "Interpretability" {
  class FeatureAttribution
  class CellAttributionMap
  class SpatialAttributionMap
  class InterpretationResult
  abstract class AbstractModelInterpreter
  class EfficientNetInterpreter
  class GNNInterpreter
  class InterpretationService
}

package "Loaders" {
  abstract class AbstractTranscriptomicsLoader
  class XeniumGenePanelLoader

  abstract class AbstractWSILoader
  class XeniumWSILoader

  abstract class AbstractCellBoundaryLoader
  class XeniumCellBoundaryAdapter

  class XeniumGraphDataset
}

package "Pipelines" {
  abstract class AbstractTrainPipeline
  class SpatialTrainPipeline
}

package "Models" {
  abstract class AbstractExpertModel
  class EfficientNetExpertAdapter
  class RadiomicsExpertAdapter
  class GeometricMorphologyExpert

  abstract class AbstractSpatialSpotModel
  class TranscriptomicsGNN
  class SpatialGNNAdapter
}

package "Data Contracts" {
  class FeatureMeta
  class ExpertModelManifest
  class CellFeatureValue
  class CellFeaturePassport
  class WSICellFeatureRegistry
  class SpotExpressionTarget
  class SpotExpressionPrediction
  class RegionalPredictionOutput

  class PolygonModel
  class SegmentationFeatureMeta
  class SegmentationModelManifest
  class SegmentationFeatureValue
  class ImageMetadata
  class SegmentationInput
  class SegmentationCellInstance
  class SegmentationContainer

  class WSIUploadRequest
  class WSILevelMetadata
  class WSIContainer
  class WSIRegionContainer
}

AbstractModelRegistry <|-- ModelPresetRegistry
BaseSpatialOrchestrator <|-- XeniumSystemOrchestrator
AbstractModelInterpreter <|-- EfficientNetInterpreter
AbstractModelInterpreter <|-- GNNInterpreter
AbstractTranscriptomicsLoader <|-- XeniumGenePanelLoader
AbstractWSILoader <|-- XeniumWSILoader
AbstractCellBoundaryLoader <|-- XeniumCellBoundaryAdapter
AbstractTrainPipeline <|-- SpatialTrainPipeline
AbstractExpertModel <|-- EfficientNetExpertAdapter
AbstractExpertModel <|-- RadiomicsExpertAdapter
AbstractExpertModel <|-- GeometricMorphologyExpert
AbstractSpatialSpotModel <|-- SpatialGNNAdapter

SpatialGNNAdapter *-- TranscriptomicsGNN
ModelPresetRegistry *-- PresetConfiguration
XeniumSystemOrchestrator --> DatabaseService
XeniumSystemOrchestrator --> ModelPresetRegistry
XeniumSystemOrchestrator --> InterpretationService
XeniumSystemOrchestrator --> XeniumWSILoader
XeniumSystemOrchestrator --> XeniumGenePanelLoader
XeniumSystemOrchestrator --> XeniumCellBoundaryAdapter
XeniumSystemOrchestrator --> SpatialTrainPipeline
XeniumSystemOrchestrator --> XeniumGraphDataset

InterpretationService --> ModelPresetRegistry
InterpretationService --> EfficientNetInterpreter
InterpretationService --> GNNInterpreter
EfficientNetInterpreter --> EfficientNetExpertAdapter
GNNInterpreter --> SpatialGNNAdapter

XeniumCellBoundaryAdapter --> XeniumWSILoader
XeniumCellBoundaryAdapter --> XeniumGenePanelLoader
SpatialTrainPipeline --> SpatialGNNAdapter
XeniumGraphDataset --> WSICellFeatureRegistry

User "1" --> "0..*" Project
User "1" --> "0..*" AnalysisReport
User "1" --> "0..*" AuditLog
Project "1" --> "0..*" XeniumSlide
XeniumSlide "1" --> "0..*" AnalysisReport
ModelPreset "1" --> "0..*" AnalysisReport
ModelPreset --> MLModel : expert_model
ModelPreset --> MLModel : spatial_model

DatabaseService ..> User
DatabaseService ..> Project
DatabaseService ..> XeniumSlide
DatabaseService ..> MLModel
DatabaseService ..> ModelPreset
DatabaseService ..> AnalysisReport
DatabaseService ..> AuditLog

ExpertModelManifest *-- FeatureMeta
CellFeaturePassport *-- SegmentationCellInstance
CellFeaturePassport *-- CellFeatureValue
WSICellFeatureRegistry *-- ExpertModelManifest
WSICellFeatureRegistry *-- CellFeaturePassport
RegionalPredictionOutput *-- SpotExpressionPrediction
SegmentationModelManifest *-- SegmentationFeatureMeta
SegmentationInput --> ImageMetadata
SegmentationCellInstance *-- PolygonModel
SegmentationCellInstance *-- SegmentationFeatureValue
SegmentationContainer *-- SegmentationCellInstance
WSIContainer *-- WSILevelMetadata
AbstractWSILoader ..> WSIContainer
AbstractWSILoader ..> WSIRegionContainer
AbstractCellBoundaryLoader ..> WSIRegionContainer
AbstractCellBoundaryLoader ..> SegmentationContainer
AbstractExpertModel ..> SegmentationContainer
AbstractExpertModel ..> WSICellFeatureRegistry
AbstractSpatialSpotModel ..> WSICellFeatureRegistry
AbstractSpatialSpotModel ..> SpotExpressionTarget
AbstractSpatialSpotModel ..> RegionalPredictionOutput
AbstractModelInterpreter ..> InterpretationResult
InterpretationResult *-- FeatureAttribution
InterpretationResult *-- CellAttributionMap
InterpretationResult *-- SpatialAttributionMap
@enduml
```

---

# 2. Диаграммы вариантов использования

## 2.1. Обзорная диаграмма вариантов использования

### Файл
`diploma/docs/uml/use-case-overview.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Clinician" as Clinician
actor "Researcher" as Researcher
actor "Administrator" as Admin

rectangle "SpatialScope" {
  usecase "Аутентификация и доступ" as UC_Access
  usecase "Проекты и слайды" as UC_ProjectSlides
  usecase "Пресеты и инференс" as UC_InferenceBlock
  usecase "Обучение" as UC_TrainingBlock
  usecase "Отчёты и интерпретация" as UC_ReportsBlock
}

Clinician --> UC_Access
Clinician --> UC_ProjectSlides
Clinician --> UC_InferenceBlock
Clinician --> UC_ReportsBlock

Researcher --> UC_Access
Researcher --> UC_ProjectSlides
Researcher --> UC_InferenceBlock
Researcher --> UC_TrainingBlock
Researcher --> UC_ReportsBlock

Admin --> UC_Access
Admin --> UC_ProjectSlides
Admin --> UC_InferenceBlock
Admin --> UC_TrainingBlock
Admin --> UC_ReportsBlock
@enduml
```

## 2.2. Пользователи и доступ

### Файл
`diploma/docs/uml/use-case-access.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Clinician" as Clinician
actor "Researcher" as Researcher
actor "Administrator" as Admin

rectangle "SpatialScope — Пользователи и доступ" {
  usecase "Войти в систему" as UC_Login
  usecase "Получить профиль\nтекущего пользователя" as UC_Profile
  usecase "Зарегистрировать\nпользователя" as UC_RegisterUser
}

Clinician --> UC_Login
Clinician --> UC_Profile

Researcher --> UC_Login
Researcher --> UC_Profile

Admin --> UC_Login
Admin --> UC_Profile
Admin --> UC_RegisterUser
@enduml
```

## 2.3. Проекты и слайды

### Файл
`diploma/docs/uml/use-case-projects-slides.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Clinician" as Clinician
actor "Researcher" as Researcher
actor "Administrator" as Admin

rectangle "SpatialScope — Проекты и слайды" {
  usecase "Просмотреть проекты" as UC_ViewProjects
  usecase "Создать проект" as UC_CreateProject
  usecase "Зарегистрировать\nXenium-слайд" as UC_RegisterSlide
  usecase "Просмотреть метаданные\nслайда" as UC_ViewSlideMeta
  usecase "Открыть изображение\nслайда" as UC_OpenSlideImage
}

Clinician --> UC_ViewProjects
Clinician --> UC_ViewSlideMeta
Clinician --> UC_OpenSlideImage

Researcher --> UC_ViewProjects
Researcher --> UC_CreateProject
Researcher --> UC_RegisterSlide
Researcher --> UC_ViewSlideMeta
Researcher --> UC_OpenSlideImage

Admin --> UC_ViewProjects
Admin --> UC_CreateProject
Admin --> UC_RegisterSlide
Admin --> UC_ViewSlideMeta
Admin --> UC_OpenSlideImage
@enduml
```

## 2.4. Пресеты и инференс

### Файл
`diploma/docs/uml/use-case-inference.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Clinician" as Clinician
actor "Researcher" as Researcher
actor "Administrator" as Admin

rectangle "SpatialScope — Пресеты и инференс" {
  usecase "Просмотреть список\nпресетов" as UC_ViewPresets
  usecase "Активировать пресет" as UC_ActivatePreset
  usecase "Запустить инференс\nпо ROI" as UC_RunInference
  usecase "Выбрать пресет\nмодели" as UC_SelectPreset
  usecase "Построить\nинтерпретацию" as UC_Interpret
  usecase "Сохранить отчёт\nанализа" as UC_SaveReport
}

Clinician --> UC_ViewPresets
Clinician --> UC_RunInference

Researcher --> UC_ViewPresets
Researcher --> UC_RunInference

Admin --> UC_ViewPresets
Admin --> UC_ActivatePreset
Admin --> UC_RunInference

UC_RunInference .> UC_SelectPreset : <<include>>
UC_RunInference .> UC_SaveReport : <<include>>
UC_Interpret .> UC_RunInference : <<extend>>
@enduml
```

## 2.5. Обучение

### Файл
`diploma/docs/uml/use-case-training.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Researcher" as Researcher
actor "Administrator" as Admin
actor "Clinician" as Clinician

rectangle "SpatialScope — Обучение" {
  usecase "Запустить фоновое\nобучение" as UC_Train
  usecase "Настроить\ntraining ROI" as UC_SetTrainingROI
  usecase "Отслеживать\nстатус job" as UC_TrackJob
}

Researcher --> UC_Train
Researcher --> UC_TrackJob

Admin --> UC_Train
Admin --> UC_TrackJob

Clinician --> UC_TrackJob

UC_Train .> UC_SetTrainingROI : <<include>>
UC_Train .> UC_TrackJob : <<include>>
@enduml
```

## 2.6. Отчёты и интерпретация

### Файл
`diploma/docs/uml/use-case-reports.puml`

```plantuml
@startuml
left to right direction
skinparam backgroundColor transparent
skinparam shadowing false
skinparam packageStyle rectangle
skinparam usecase {
  BackgroundColor #FFFFFF
  BorderColor #2c3e50
  ArrowColor #34495e
  FontColor #2c3e50
}
skinparam actor {
  BorderColor #2c3e50
  BackgroundColor #ecf0f1
  FontColor #2c3e50
}

actor "Clinician" as Clinician
actor "Researcher" as Researcher
actor "Administrator" as Admin

rectangle "SpatialScope — Отчёты и интерпретация" {
  usecase "Просмотреть отчёты\nпо слайду" as UC_ViewSlideReports
  usecase "Открыть отчёт" as UC_OpenReport
  usecase "Интерпретировать отчёт" as UC_InterpretReport
  usecase "Экспортировать отчёт\nв JSON" as UC_ExportReport
}

Clinician --> UC_ViewSlideReports
Clinician --> UC_OpenReport
Clinician --> UC_InterpretReport
Clinician --> UC_ExportReport

Researcher --> UC_ViewSlideReports
Researcher --> UC_OpenReport
Researcher --> UC_InterpretReport
Researcher --> UC_ExportReport

Admin --> UC_ViewSlideReports
Admin --> UC_OpenReport
Admin --> UC_InterpretReport
Admin --> UC_ExportReport
@enduml
```

---

# 3. Диаграммы последовательности

## 3.1. Аутентификация

### Файл
`diploma/docs/uml/sequence-login.puml`

```plantuml
@startuml
actor User as U
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/users/login" as API
participant "DatabaseService" as DB
participant "JWT Service" as JWT

U -> FE : Ввод email и password
FE -> API : POST /api/users/login
API -> DB : get_user_by_email(email)
DB --> API : User
API -> API : verify_password(password, user.password_hash)
API -> JWT : create_access_token({sub, email, role})
JWT --> API : token
API --> FE : success + token + user
FE --> U : Сохранение токена и переход в dashboard
@enduml
```

## 3.2. Создание проекта

### Файл
`diploma/docs/uml/sequence-create-project.puml`

```plantuml
@startuml
actor Researcher as R
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/projects" as API
participant "XeniumSystemOrchestrator" as ORCH
participant "DatabaseService" as DB

R -> FE : Ввод названия и описания проекта
FE -> API : POST /api/projects
API -> ORCH : create_new_project(user_id, name, description, ip_address)
ORCH -> DB : create_project(...)
DB --> ORCH : Project
ORCH -> DB : log_action(... CREATE_PROJECT ...)
ORCH --> API : project_id
API --> FE : success + project_id
FE --> R : Отображение нового проекта в dashboard
@enduml
```

## 3.3. Регистрация слайда

### Файл
`diploma/docs/uml/sequence-register-slide.puml`

```plantuml
@startuml
actor Researcher as R
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/slides/register" as API
participant "XeniumSystemOrchestrator" as ORCH
participant "XeniumGenePanelLoader" as GENE
participant "XeniumWSILoader" as WSI
participant "XeniumCellBoundaryAdapter" as BOUND
participant "DatabaseService" as DB

R -> FE : Указывает project_id, patient_name, пути к TIFF/Parquet/HDF5
FE -> API : POST /api/slides/register
API -> ORCH : register_and_cache_slide(...)
ORCH -> ORCH : Проверка существования файлов
ORCH -> GENE : new XeniumGenePanelLoader(matrix_path)
ORCH -> WSI : new XeniumWSILoader(cells_path)
ORCH -> BOUND : new XeniumCellBoundaryAdapter(wsi_loader, gene_loader)
ORCH -> WSI : load_slide_metadata(tiff_path)
WSI --> ORCH : WSIContainer
ORCH -> WSI : apply_registration_to_dataframe()
ORCH -> DB : register_xenium_slide(..., metadata)
DB --> ORCH : XeniumSlide
ORCH -> ORCH : slides_cache[slide_id] = {loaders, metadata, db_record}
ORCH --> API : slide_id
API --> FE : success + slide_id
@enduml
```

## 3.4. Просмотр слайда

### Файл
`diploma/docs/uml/sequence-view-slide.puml`

```plantuml
@startuml
actor Clinician as C
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/slides/{slide_id}" as META
participant "FastAPI /api/slides/{slide_id}/image" as IMG
participant "XeniumSystemOrchestrator" as ORCH
participant "Filesystem" as FS

C -> FE : Открывает страницу слайда
FE -> META : GET /api/slides/{slide_id}
META -> ORCH : get_slide_metadata(slide_id)
ORCH --> META : metadata + paths + status
META --> FE : success + metadata
FE -> IMG : GET /api/slides/{slide_id}/image
IMG -> ORCH : get_slide_metadata(slide_id)
ORCH --> IMG : tiff_path
IMG -> FS : Проверка наличия TIFF
FS --> IMG : Файл доступен
IMG --> FE : FileResponse(image/tiff)
FE --> C : Отображение метаданных и изображения
@enduml
```

## 3.5. Инференс

### Файл
`diploma/docs/uml/sequence-run-inference.puml`

```plantuml
@startuml
actor Clinician as C
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/inference/run" as API
participant "XeniumSystemOrchestrator" as ORCH
participant "ModelPresetRegistry" as REG
participant "XeniumWSILoader" as WSI
participant "XeniumCellBoundaryAdapter" as BOUND
participant "EfficientNetExpertAdapter / RadiomicsExpertAdapter / GeometricMorphologyExpert" as EXP
participant "SpatialGNNAdapter" as GNN
participant "InterpretationService" as IS
participant "EfficientNetInterpreter" as EINT
participant "GNNInterpreter" as GINT
participant "DatabaseService" as DB

C -> FE : Выбор ROI, patch_size, preset, target_gene
FE -> API : POST /api/inference/run
API -> ORCH : run_inference_and_save_report(...)
ORCH -> REG : get_preset(preset_id or active)
ORCH -> ORCH : _ensure_slide_is_cached(slide_id)
ORCH -> REG : get_models_for_preset(preset_id)
REG --> ORCH : concrete expert models + SpatialGNNAdapter
ORCH -> REG : get_weights_path(preset_id)
ORCH -> GNN : load_weights(path)
ORCH -> WSI : load_region(metadata, x, y, patch_size, patch_size)
WSI --> ORCH : WSIRegionContainer
ORCH -> BOUND : load_boundaries(patch_region,...)
BOUND --> ORCH : SegmentationContainer
ORCH -> EXP : predict(...)
EXP --> ORCH : WSICellFeatureRegistry
ORCH -> GNN : predict_spatial_expression(registry, regional_spots, patch_id)
GNN --> ORCH : RegionalPredictionOutput
opt include_interpretation = true
  ORCH -> IS : interpret_patch(...)
  IS -> EINT : interpret(...)
  IS -> GINT : interpret(...)
  EINT --> IS : InterpretationResult
  GINT --> IS : InterpretationResult
  IS --> ORCH : List<InterpretationResult>
end
ORCH -> DB : create_analysis_report(...)
ORCH -> DB : log_action(... RUN_INFERENCE ...)
DB --> ORCH : AnalysisReport
ORCH --> API : report_id + predicted_expression + interpretation
API --> FE : success
@enduml
```

## 3.6. Фоновое обучение

### Файл
`diploma/docs/uml/sequence-start-training.puml`

```plantuml
@startuml
actor Researcher as R
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/training/start" as API
participant "XeniumSystemOrchestrator" as ORCH
participant "XeniumWSILoader" as WSI
participant "XeniumCellBoundaryAdapter" as BOUND
participant "EfficientNetExpertAdapter / RadiomicsExpertAdapter / GeometricMorphologyExpert" as EXP
participant "XeniumGraphDataset" as DATASET
participant "SpatialTrainPipeline" as TRAIN
participant "SpatialGNNAdapter" as GNN
participant "DatabaseService" as DB

R -> FE : Задаёт training ROI и параметры обучения
FE -> API : POST /api/training/start
API -> ORCH : start_background_training(...)
ORCH -> ORCH : jobs_registry[job_id] = running
ORCH -> DB : update_slide_status(slide_id, processing)
ORCH -> DB : log_action(... START_TRAINING ...)
ORCH --> API : job_id
API --> FE : success + job_id

loop background worker
  ORCH -> WSI : load_region(...)
  ORCH -> BOUND : load_boundaries(...)
  ORCH -> EXP : predict(...)
  EXP --> ORCH : WSICellFeatureRegistry
end

ORCH -> DATASET : new XeniumGraphDataset(registries, target_gene_names, all_gene_names, expert_key)
ORCH -> TRAIN : new SpatialTrainPipeline(dataset, GNN, optimizer,...)
TRAIN -> TRAIN : data_prepare()
TRAIN -> GNN : train forward/backward epochs
TRAIN -> TRAIN : _save_checkpoint()
TRAIN --> ORCH : training complete
ORCH -> DB : update_slide_status(slide_id, ready)
ORCH -> DB : log_action(... TRAINING_FINISHED ...)
ORCH -> ORCH : jobs_registry[job_id] = success
@enduml
```

## 3.7. Открытие отчёта

### Файл
`diploma/docs/uml/sequence-open-report.puml`

```plantuml
@startuml
actor Clinician as C
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/reports/slide/{slide_id}" as LIST
participant "FastAPI /api/reports/{report_id}" as DETAIL
participant "DatabaseService" as DB

C -> FE : Открывает раздел отчётов слайда
FE -> LIST : GET /api/reports/slide/{slide_id}
LIST -> DB : get_slide_reports(slide_id)
DB --> LIST : List<AnalysisReport>
LIST --> FE : success + report summaries
C -> FE : Выбирает отчёт
FE -> DETAIL : GET /api/reports/{report_id}
DETAIL -> DB : get_report_by_id(report_id)
DB --> DETAIL : AnalysisReport
DETAIL --> FE : success + detailed report
FE --> C : Отображение полного отчёта
@enduml
```

## 3.8. Интерпретация отчёта

### Файл
`diploma/docs/uml/sequence-interpret-report.puml`

```plantuml
@startuml
actor Clinician as C
participant "SpatialScope Frontend" as FE
participant "FastAPI /api/reports/{report_id}/interpret" as API
participant "XeniumSystemOrchestrator" as ORCH
participant "InterpretationService" as IS
participant "EfficientNetInterpreter" as EINT
participant "GNNInterpreter" as GINT
participant "DatabaseService" as DB

C -> FE : Запрашивает интерпретацию отчёта
FE -> API : POST /api/reports/{report_id}/interpret
API -> ORCH : interpret_report(report_id, target_gene)
ORCH -> DB : get_report_by_id(report_id)
DB --> ORCH : AnalysisReport
ORCH -> IS : interpret_saved_report(...)
IS -> EINT : interpret(...)
IS -> GINT : interpret(...)
EINT --> IS : InterpretationResult
GINT --> IS : InterpretationResult
IS --> ORCH : List<InterpretationResult>
ORCH --> API : interpreted report data
API --> FE : success
FE --> C : Отображение интерпретации
@enduml
```

---

# 4. Диаграмма деятельности

## Файл
`diploma/docs/uml/activity-diagram.puml`

```plantuml
@startuml
start
:Пользователь проходит аутентификацию;
:Frontend получает роль и открывает dashboard;
if (Нужно зарегистрировать слайд?) then (Да)
  :Researcher/Admin отправляет пути к TIFF, Parquet, HDF5;
  :XeniumSystemOrchestrator создаёт XeniumGenePanelLoader;
  :XeniumSystemOrchestrator создаёт XeniumWSILoader;
  :XeniumSystemOrchestrator создаёт XeniumCellBoundaryAdapter;
  :Загружаются WSI metadata и кэш слайда;
  :Слайд регистрируется в БД;
endif
:Пользователь открывает слайд;
if (Запущен инференс?) then (Да)
  :Выбор ROI и активного пресета;
  :Извлечение patch через XeniumWSILoader;
  :Сегментация/границы через XeniumCellBoundaryAdapter;
  :Извлечение экспертных признаков concrete expert adapter;
  :Прогон SpatialGNNAdapter;
  :Построение интерпретации через InterpretationService;
  :Сохранение AnalysisReport и AuditLog;
else (Обучение)
  :Выбор training ROI и гиперпараметров;
  :Сбор WSICellFeatureRegistry по патчам;
  :Формирование XeniumGraphDataset;
  :Запуск SpatialTrainPipeline;
  :Сохранение checkpoint и обновление статусов;
endif
:Просмотр и экспорт отчёта;
stop
@enduml
```

---

# 5. Диаграмма компонентов

## Файл
`diploma/docs/uml/component-diagram.puml`

```plantuml
@startuml
skinparam componentStyle rectangle

actor "Administrator" as Admin
actor "Researcher" as Researcher
actor "Clinician" as Clinician

component "SpatialScope Frontend\n(Vue 3 + Vite)" as Frontend
component "REST API\nFastAPI app.main" as API
component "Auth & Role Guard\nJWT / require_roles" as Auth
component "XeniumSystemOrchestrator" as Orchestrator
component "ModelPresetRegistry" as Registry
component "InterpretationService" as Interpret
component "DatabaseService" as Persistence
component "XeniumWSILoader" as WSILoader
component "XeniumGenePanelLoader" as RNALoader
component "XeniumCellBoundaryAdapter" as Boundary
component "EfficientNetExpertAdapter" as EffNet
component "RadiomicsExpertAdapter" as Radio
component "GeometricMorphologyExpert" as Geom
component "SpatialGNNAdapter" as GNN
component "EfficientNetInterpreter" as EInt
component "GNNInterpreter" as GInt
component "XeniumGraphDataset" as Dataset
component "SpatialTrainPipeline" as Train
component "Filesystem\nTIFF / Parquet / HDF5 / checkpoints" as FS
component "SQLite database" as DB

Admin --> Frontend
Researcher --> Frontend
Clinician --> Frontend
Frontend --> API
API --> Auth
API --> Orchestrator
API --> Registry
API --> Persistence

Orchestrator --> Registry
Orchestrator --> Persistence
Orchestrator --> Interpret
Orchestrator --> WSILoader
Orchestrator --> RNALoader
Orchestrator --> Boundary
Orchestrator --> EffNet
Orchestrator --> Radio
Orchestrator --> Geom
Orchestrator --> GNN
Orchestrator --> Dataset
Orchestrator --> Train

Interpret --> Registry
Interpret --> EInt
Interpret --> GInt
EInt --> EffNet
GInt --> GNN

WSILoader --> FS
RNALoader --> FS
Boundary --> FS
GNN --> FS
Train --> FS
Persistence --> DB
@enduml
```

---

# 6. Диаграмма развёртывания

## Файл
`diploma/docs/uml/deployment-diagram.puml`

```plantuml
@startuml
node "Client Device" {
  artifact "Web Browser" as Browser
  artifact "SpatialScope Frontend\n(Vue 3 + Vite SPA)" as Frontend
}

node "Application Server" {
  artifact "FastAPI REST API\napp.main" as API
  artifact "Auth & Role Guard\nJWT / require_roles" as Auth
  artifact "XeniumSystemOrchestrator" as Orchestrator
  artifact "ModelPresetRegistry" as Registry
  artifact "InterpretationService" as Interpret
  artifact "Expert Models Runtime\nEfficientNet / Radiomics / Geometry" as Experts
  artifact "Spatial Runtime\nSpatialGNNAdapter" as GNN
  artifact "Training Runtime\nXeniumGraphDataset + SpatialTrainPipeline" as Train
}

node "Data Storage" {
  database "SQLite DB\nusers, projects, slides, reports, audit" as DB
  folder "Filesystem\nTIFF / Parquet / HDF5" as DataFiles
  folder "Filesystem\ncache / checkpoints" as CacheFiles
}

Browser --> Frontend : HTTPS / UI rendering
Frontend --> API : HTTP(S) JSON API
API --> Auth : token validation
API --> Orchestrator : application calls
API --> Registry : preset listing / activation
API --> DB : persistence operations
Orchestrator --> Registry : resolve models
Orchestrator --> Interpret : build explanations
Orchestrator --> Experts : feature extraction
Orchestrator --> GNN : inference
Orchestrator --> Train : training jobs
Orchestrator --> DataFiles : load TIFF / Parquet / HDF5
Experts --> CacheFiles : embeddings cache
GNN --> CacheFiles : weights load/save
Train --> CacheFiles : checkpoints
@enduml
```

---

# 7. Диаграммы состояний

## 7.1. User

### Файл
`diploma/docs/uml/state/User.puml`

```plantuml
@startuml
[*] --> Registered
Registered --> Authenticated : login()
Authenticated --> AuthorizedAdmin : role == admin
Authenticated --> AuthorizedResearcher : role == researcher
Authenticated --> AuthorizedClinician : role == clinician
AuthorizedAdmin --> Authenticated : token refresh / profile reload
AuthorizedResearcher --> Authenticated : token refresh / profile reload
AuthorizedClinician --> Authenticated : token refresh / profile reload
Authenticated --> LoggedOut : logout() / token expired
LoggedOut --> Authenticated : login()
@enduml
```

## 7.2. Project

### Файл
`diploma/docs/uml/state/Project.puml`

```plantuml
@startuml
[*] --> Created
Created --> VisibleInDashboard : get_dashboard_data()
VisibleInDashboard --> ContainsSlides : register_and_cache_slide()
ContainsSlides --> ContainsSlides : get_project_slides()
ContainsSlides --> VisibleInDashboard : project has no active slides / cleanup
@enduml
```

## 7.3. XeniumSlide

### Файл
`diploma/docs/uml/state/XeniumSlide.puml`

```plantuml
@startuml
[*] --> PathsProvided
PathsProvided --> FilesValidated : register_and_cache_slide() / files exist
FilesValidated --> MetadataLoaded : load_slide_metadata()
MetadataLoaded --> Cached : slides_cache[slide_id] initialized
Cached --> Ready : register_xenium_slide(status=ready)
Ready --> InferenceAccessible : get_slide_metadata() / load_region()
InferenceAccessible --> Ready
Ready --> TrainingProcessing : start_background_training()
TrainingProcessing --> Ready : update_slide_status(ready)
TrainingProcessing --> Failed : update_slide_status(failed)
Failed --> TrainingProcessing : retry training
@enduml
```

## 7.4. MLModel

### Файл
`diploma/docs/uml/state/MLModel.puml`

```plantuml
@startuml
[*] --> Declared
Declared --> RegisteredInRegistry : register_expert() / register_spatial_model()
RegisteredInRegistry --> BoundToPreset : create_preset()
BoundToPreset --> LoadedForExecution : get_models_for_preset() / load_weights()
LoadedForExecution --> BoundToPreset : inference or training finished
@enduml
```

## 7.5. ModelPreset

### Файл
`diploma/docs/uml/state/ModelPreset.puml`

```plantuml
@startuml
[*] --> Defined
Defined --> StoredInSystem : create_preset()
StoredInSystem --> AvailableForSelection : get_available_presets()
AvailableForSelection --> Active : set_active_preset()
Active --> UsedForInference : run_inference_and_save_report()
Active --> UsedForTraining : start_background_training()
UsedForInference --> Active
UsedForTraining --> Active
Active --> AvailableForSelection : another preset activated
@enduml
```

## 7.6. AnalysisReport

### Файл
`diploma/docs/uml/state/AnalysisReport.puml`

```plantuml
@startuml
[*] --> PredictionGenerated
PredictionGenerated --> InterpretationAttached : interpret_patch()
InterpretationAttached --> Persisted : create_analysis_report()
Persisted --> Listed : get_slide_reports()
Persisted --> Opened : get_report_by_id()
Opened --> Exported : export_report_json()
Exported --> Opened
Listed --> Opened : select report
@enduml
```

## 7.7. AuditLog

### Файл
`diploma/docs/uml/state/AuditLog.puml`

```plantuml
@startuml
[*] --> EventRaised
EventRaised --> Persisted : log_action()
Persisted --> AvailableForAudit : query / review history
AvailableForAudit --> AvailableForAudit : append next events
@enduml
```

## 7.8. DatabaseService

### Файл
`diploma/docs/uml/state/DatabaseService.puml`

```plantuml
@startuml
[*] --> Idle
Idle --> SessionOpened : _execute()
SessionOpened --> Reading : get_* operation
SessionOpened --> Writing : create_* / update_* / log_action()
Reading --> Committed : success
Writing --> Committed : success
Reading --> RolledBack : exception
Writing --> RolledBack : exception
Committed --> SessionClosed : session.close()
RolledBack --> SessionClosed : session.close()
SessionClosed --> Idle
@enduml
```

## 7.9. XeniumSystemOrchestrator

### Файл
`diploma/docs/uml/state/XeniumSystemOrchestrator.puml`

```plantuml
@startuml
[*] --> Ready
Ready --> SlideRegistration : register_and_cache_slide()
SlideRegistration --> Ready : slide cached and persisted
Ready --> DashboardAggregation : get_dashboard_data()
DashboardAggregation --> Ready
Ready --> InferenceExecution : run_inference_and_save_report()
InferenceExecution --> ReportPersistence : create_analysis_report()
ReportPersistence --> Ready
Ready --> TrainingLaunch : start_background_training()
TrainingLaunch --> JobMonitoring : get_job_status()
JobMonitoring --> Ready : success
JobMonitoring --> FailureHandling : failed
FailureHandling --> Ready : retry / recovery
Ready --> ReportInterpretation : interpret_report()
ReportInterpretation --> Ready
@enduml
```

## 7.10. ModelPresetRegistry

### Файл
`diploma/docs/uml/state/ModelPresetRegistry.puml`

```plantuml
@startuml
[*] --> Empty
Empty --> ExpertsRegistered : register_expert()
ExpertsRegistered --> SpatialModelsRegistered : register_spatial_model()
SpatialModelsRegistered --> PresetsAvailable : create_preset()
PresetsAvailable --> ActivePresetSelected : set_active_preset()
ActivePresetSelected --> RuntimeResolved : get_models_for_preset()
RuntimeResolved --> ActivePresetSelected
ActivePresetSelected --> PresetsEnumerated : get_available_presets()
PresetsEnumerated --> ActivePresetSelected
@enduml
```

## 7.11. PresetConfiguration

### Файл
`diploma/docs/uml/state/PresetConfiguration.puml`

```plantuml
@startuml
[*] --> Defined
Defined --> Validated : create_preset() / IDs resolved in registry
Validated --> Listed : get_available_presets()
Listed --> Selected : set_active_preset()
Selected --> Resolved : get_models_for_preset()
Resolved --> Selected
@enduml
```

## 7.12. InterpretationService

### Файл
`diploma/docs/uml/state/InterpretationService.puml`

```plantuml
@startuml
[*] --> Idle
Idle --> InterpretingPatch : interpret_patch()
InterpretingPatch --> EfficientNetPass : EfficientNet expert present
InterpretingPatch --> GNNPass : spatial model present
EfficientNetPass --> MergeResults
GNNPass --> MergeResults
MergeResults --> Returned
Returned --> Idle
@enduml
```

## 7.13. XeniumWSILoader

### Файл
`diploma/docs/uml/state/XeniumWSILoader.puml`

```plantuml
@startuml
[*] --> ParquetLoaded
ParquetLoaded --> BoundariesLoaded : read boundary parquet
BoundariesLoaded --> CoordinatesNormalized : _normalize_cell_id() / apply_registration_to_dataframe()
CoordinatesNormalized --> MetadataReady : load_slide_metadata()
MetadataReady --> RegionExtractionReady : load_region()
RegionExtractionReady --> CellQueryReady : get_cells_in_region()
CellQueryReady --> RegionExtractionReady
@enduml
```

## 7.14. XeniumGenePanelLoader

### Файл
`diploma/docs/uml/state/XeniumGenePanelLoader.puml`

```plantuml
@startuml
[*] --> MatrixOpened
MatrixOpened --> MatrixIndexed : HDF5 matrix loaded in memory
MatrixIndexed --> GeneNamesAvailable : genes catalog prepared
GeneNamesAvailable --> VectorQueryAvailable : get_gene_vector()
VectorQueryAvailable --> GeneNamesAvailable : repeated queries
@enduml
```

## 7.15. XeniumCellBoundaryAdapter

### Файл
`diploma/docs/uml/state/XeniumCellBoundaryAdapter.puml`

```plantuml
@startuml
[*] --> Ready
Ready --> ManifestExposed : manifest
ManifestExposed --> Ready
Ready --> PatchCellsCollected : load_boundaries() / get_cells_in_region()
PatchCellsCollected --> GeneVectorsAttached : get_gene_vector()
GeneVectorsAttached --> CoordinatesWarpedOrShifted : registration or patch offset applied
CoordinatesWarpedOrShifted --> SegmentationBuilt : SegmentationContainer returned
SegmentationBuilt --> Ready
@enduml
```

## 7.16. EfficientNetExpertAdapter

### Файл
`diploma/docs/uml/state/EfficientNetExpertAdapter.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> CacheChecked : predict()
CacheChecked --> LoadedFromCache : cached registry exists
CacheChecked --> PatchPrepared : cache miss
PatchPrepared --> ROIsExtracted : roi_align()
ROIsExtracted --> EmbeddingsComputed : feature_extractor()
EmbeddingsComputed --> RegistryBuilt : WSICellFeatureRegistry
RegistryBuilt --> SavedToCache : pickle.dump()
SavedToCache --> ReadyForReuse
LoadedFromCache --> ReadyForReuse
ReadyForReuse --> CacheChecked : next predict()
@enduml
```

## 7.17. RadiomicsExpertAdapter

### Файл
`diploma/docs/uml/state/RadiomicsExpertAdapter.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> CacheChecked : predict()
CacheChecked --> LoadedFromCache : cached registry exists
CacheChecked --> PatchConverted : grayscale / masks prepared
PatchConverted --> RadiomicsComputed : extractor.execute()
RadiomicsComputed --> RegistryBuilt : WSICellFeatureRegistry
RegistryBuilt --> SavedToCache : pickle.dump()
SavedToCache --> ReadyForReuse
LoadedFromCache --> ReadyForReuse
@enduml
```

## 7.18. GeometricMorphologyExpert

### Файл
`diploma/docs/uml/state/GeometricMorphologyExpert.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> CacheChecked : predict()
CacheChecked --> LoadedFromCache : cached registry exists
CacheChecked --> NeighborhoodBuilt : KDTree / centroids prepared
NeighborhoodBuilt --> MorphologyComputed : polygons processed
MorphologyComputed --> RegistryBuilt : WSICellFeatureRegistry
RegistryBuilt --> SavedToCache : pickle.dump()
SavedToCache --> ReadyForReuse
LoadedFromCache --> ReadyForReuse
@enduml
```

## 7.19. SpatialGNNAdapter

### Файл
`diploma/docs/uml/state/SpatialGNNAdapter.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> WeightsLoaded : load_weights()
Initialized --> GraphReady : _build_graph()
WeightsLoaded --> GraphReady
GraphReady --> InferenceRunning : predict_spatial_expression()
InferenceRunning --> PredictionReturned : RegionalPredictionOutput
PredictionReturned --> GraphReady
GraphReady --> Saved : save_weights()
Saved --> GraphReady
@enduml
```

## 7.20. EfficientNetInterpreter

### Файл
`diploma/docs/uml/state/EfficientNetInterpreter.puml`

```plantuml
@startuml
[*] --> Ready
Ready --> HooksRegistered : interpret()
HooksRegistered --> ForwardExecuted : feature_extractor forward
ForwardExecuted --> BackwardExecuted : score.backward()
BackwardExecuted --> CAMBuilt : GradCAM map computed
CAMBuilt --> ResultBuilt : InterpretationResult
ResultBuilt --> Ready
@enduml
```

## 7.21. GNNInterpreter

### Файл
`diploma/docs/uml/state/GNNInterpreter.puml`

```plantuml
@startuml
[*] --> Ready
Ready --> GraphPrepared : interpret() / _build_graph()
GraphPrepared --> AttentionCollected : conv1(... return_attention_weights=True)
AttentionCollected --> GradientsComputed : backward()
GradientsComputed --> NodeScoresCombined : gradients + attention merged
NodeScoresCombined --> ResultBuilt : InterpretationResult
ResultBuilt --> Ready
@enduml
```

## 7.22. XeniumGraphDataset

### Файл
`diploma/docs/uml/state/XeniumGraphDataset.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> PreparingGraphs : _prepare_data()
PreparingGraphs --> GraphSkipped : too few cells / invalid registry
PreparingGraphs --> GraphBuilt : Data(x, edge_index, y)
GraphBuilt --> Stored : append to _precomputed_graphs
GraphSkipped --> PreparingGraphs : next registry
Stored --> Ready : all registries processed
Ready --> Accessed : __getitem__() / __len__()
Accessed --> Ready
@enduml
```

## 7.23. SpatialTrainPipeline

### Файл
`diploma/docs/uml/state/SpatialTrainPipeline.puml`

```plantuml
@startuml
[*] --> Initialized
Initialized --> DataPrepared : data_prepare()
DataPrepared --> TrainingRunning : train()
TrainingRunning --> EpochCompleted : loss computed
EpochCompleted --> CheckpointSaved : _save_checkpoint()
CheckpointSaved --> TrainingRunning : next epoch
EpochCompleted --> BestModelUpdated : loss < best_loss
BestModelUpdated --> TrainingRunning
TrainingRunning --> Finished : all epochs complete
Finished --> BestModelLoaded : load_best_model()
BestModelLoaded --> Finished
@enduml
```

---

## Примечание

Если понадобится, следующий шаг можно сделать таким:

1. собрать ещё и **единый документ с текстовыми описаниями + PlantUML-кодом**;
2. или отдельно сформировать **только state-диаграммы** в самостоятельный отчёт;
3. или подготовить **версию этого файла, оптимизированную для вставки в Word**.
