# System Architecture

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA                                  │
│  • beniplus_BE_git_log_*.csv / _full_diff.txt                   │
│  • beniplus_FE_git_log_*.csv / _full_diff.txt                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  STEP 1: GitLogParser               │
        │  • Parse CSV + diff files           │
        │  • Create Commit objects            │
        │  • Return: List[Commit]             │
        │  LOG: Commits per repo              │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 2: EmbeddingGenerator         │
        │  ┌──────────────────────────────┐  │
        │  │ CacheManager checks cache    │  │
        │  │  cached? YES → load from CSV │  │
        │  │  cached? NO → API call       │  │
        │  └──────────────────────────────┘  │
        │  • Generate vectors for commits    │
        │  • Batch API calls (100 at a time)│
        │  • Save to cache_embeddings.csv    │
        │  • Return: np.ndarray (2230xD)     │
        │  LOG: Cache hits/misses, batches   │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 3: Clusterer (HDBSCAN)       │
        │  • Calculate pairwise distances    │
        │  • Identify commit groups          │
        │  • Return: List[Cluster]           │
        │  LOG: Clusters found               │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 4: TemporalValidator         │
        │  • Split clusters > 60 days        │
        │  • Preserve commit ordering        │
        │  • Return: List[Cluster]           │
        │  LOG: Clusters after validation    │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 5: SREDSignalDetector        │
        │  • Detect reverts                  │
        │  • Detect hotfix cascades          │
        │  • Detect file churn               │
        │  • Calculate SRED scores           │
        │  • Return: List[Cluster]           │
        │  LOG: Top 5 clusters by score      │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 6: LLM Generation            │
        │  ┌────────────────────────────┐   │
        │  │ LLMNarrativeGenerator      │   │
        │  │ CacheManager checks cache  │   │
        │  │  cached? YES → load        │   │
        │  │  cached? NO → GPT-4 call   │   │
        │  │ Save to cache_llm_*.csv    │   │
        │  └────────────────────────────┘   │
        │  • Generate cluster narratives    │
        │  • Generate questionnaires        │
        │  • Return: Dict[narratives]       │
        │  LOG: Progress [1/15], cache hits │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  STEP 7: Report Generation        │
        │  • Compile top 15 narratives      │
        │  • Create markdown report         │
        │  • Create JSON export             │
        │  LOG: Final report saved          │
        └────────────┬───────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT FILES                               │
│  ✓ sred_analyzer.log          (Detailed execution log)          │
│  ✓ cache_embeddings.csv       (Embedding vectors cache)         │
│  ✓ cache_llm_results.csv      (LLM results cache)               │
│  ✓ sred_report.md             (Main report, top 15)             │
│  ✓ questionnaires.md          (Knowledge gap questions)         │
│  ✓ clusters.json              (Full cluster data)               │
└─────────────────────────────────────────────────────────────────┘
```

## Caching Architecture

### Embedding Cache

```
cache_embeddings.csv
├─ Row 1: [commit_hash1, [0.001, 0.002, ..., -0.003]]
├─ Row 2: [commit_hash2, [0.005, 0.006, ..., 0.002]]
└─ Row N: [commit_hashN, [...]]

First Run:
  for each commit:
    if commit_hash NOT in cache:
      api_response = openai.embeddings.create()
      embeddings.append(api_response)
      cache[commit_hash] = api_response
  save_to_csv(cache)

Second Run:
  cache = load_from_csv()  ← All entries here
  for each commit:
    if commit_hash in cache:
      embeddings.append(cache[commit_hash])  ← Cache hit ✓
    else:
      api_response = openai.embeddings.create()
      cache[commit_hash] = api_response
  save_to_csv(cache)  ← Only if new commits added
```

### LLM Cache

```
cache_llm_results.csv
├─ Row 1: [cluster_id_0, "narrative", {...json narrative...}]
├─ Row 2: [cluster_id_0, "questionnaire", "Q1. ...?"]
├─ Row 3: [cluster_id_1, "narrative", {...json narrative...}]
└─ Row N: [cluster_id_N, "questionnaire", "Q1. ...?"]

First Run:
  for each cluster:
    if cluster_id NOT in cache:
      narrative = gpt4.generate_narrative()
      questionnaire = gpt4.generate_questionnaire()
      cache[cluster_id] = {
        'narrative': narrative,
        'questionnaire': questionnaire
      }
  save_to_csv(cache)

Second Run:
  cache = load_from_csv()  ← All entries here
  for each cluster:
    if cluster_id in cache:
      narrative = cache[cluster_id]['narrative']  ← Cache hit ✓
      questionnaire = cache[cluster_id]['questionnaire']
    else:
      narrative = gpt4.generate_narrative()
      questionnaire = gpt4.generate_questionnaire()
      cache[cluster_id] = {...}
  save_to_csv(cache)  ← Only if new clusters
```

## Class Hierarchy

```
CacheManager
├─ load_embeddings_cache() → Dict[hash, embedding]
├─ save_embeddings_cache() → None
├─ load_llm_cache() → Dict[cluster_id, Dict[type, data]]
└─ save_llm_cache() → None

GitLogParser
├─ parse_csv() → Dict[hash, metadata]
├─ parse_diff_file() → Dict[hash, diff]
└─ parse() → List[Commit]

EmbeddingGenerator
├─ cache_manager: CacheManager
├─ cache: Dict[hash, embedding]
├─ build_commit_text() → str
├─ _extract_key_snippets() → str
└─ generate_embeddings() → np.ndarray

Cluster
├─ cluster_id: int
├─ commits: List[Commit]
├─ start_date: datetime
├─ end_date: datetime
├─ authors: List[str]
├─ sred_score: float
├─ signals: Dict[str, any]
└─ span_days() → int

Clusterer
├─ min_cluster_size: int
└─ cluster() → List[Cluster]

TemporalValidator
├─ MAX_SPAN_DAYS: int = 60
└─ validate_and_split() → List[Cluster]

SREDSignalDetector
├─ REVERT_KEYWORDS: List[str]
├─ FIX_KEYWORDS: List[str]
└─ detect_signals() → Cluster

LLMNarrativeGenerator
├─ cache_manager: CacheManager
├─ cache: Dict[int, Dict]
├─ generate_narrative() → Dict[str, str]
├─ _format_commits() → str
└─ _format_signals() → str

QuestionnaireGenerator
├─ cache_manager: CacheManager
├─ cache: Dict[int, Dict]
└─ generate_questionnaire() → str
```

## Logging Architecture

```
Python Logger (logger)
├─ Level: INFO (minimum)
├─ Handlers:
│  ├─ FileHandler
│  │  └─ Output: sred_analyzer.log
│  │     Format: [timestamp] [level] [message]
│  │     Examples:
│  │     ├─ 2024-12-02 10:15:30,123 - INFO - STEP 1: Parsing git logs...
│  │     ├─ 2024-12-02 10:15:35,456 - INFO - Parsed 1250 backend commits
│  │     ├─ 2024-12-02 10:15:41,001 - INFO - Found 1500 cached embeddings
│  │     ├─ 2024-12-02 10:15:41,002 - INFO - Generating batch 1/8...
│  │     └─ 2024-12-02 10:15:41,003 - INFO - Saved embeddings cache
│  │
│  └─ StreamHandler
│     └─ Output: Console (stdout)
│        Format: Same as file
│        Real-time display

Typical Execution Log:
┌─────────────────────────────────────────────────────────────┐
│ ============================================================ │
│ 2024-12-02 10:15:30 - INFO - SRED PIPELINE STARTED         │
│ ============================================================ │
│                                                              │
│ ============================================================ │
│ 2024-12-02 10:15:31 - INFO - STEP 1: Parsing git logs...   │
│ ============================================================ │
│ 2024-12-02 10:15:35 - INFO - Parsed 1250 backend commits   │
│ 2024-12-02 10:15:40 - INFO - Parsed 980 frontend commits   │
│ 2024-12-02 10:15:40 - INFO - Total commits: 2230           │
│                                                              │
│ ============================================================ │
│ 2024-12-02 10:15:41 - INFO - STEP 2: Generating embeddings │
│ ============================================================ │
│ 2024-12-02 10:15:41 - INFO - Checking cache for 2230 commits │
│ 2024-12-02 10:15:41 - INFO - Found 1500 cached embeddings, need │
│                              to generate 730 new ones       │
│ 2024-12-02 10:15:42 - INFO - Generating batch 1/8...       │
│ 2024-12-02 10:15:45 - INFO - Generating batch 2/8...       │
│ ...                                                          │
│ 2024-12-02 10:16:10 - INFO - Saving 2230 embeddings to CSV │
│ 2024-12-02 10:16:11 - INFO - Generated all 2230 embeddings │
│ ...                                                          │
│ ============================================================ │
│ 2024-12-02 10:20:45 - INFO - PIPELINE COMPLETE!             │
│ ============================================================ │
│ 2024-12-02 10:20:45 - INFO - Check sred_analyzer.log for... │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimization

### First Run (Cold Start)
```
Embeddings:  2230 API calls (batches of 100) ~3 minutes
LLM:        15 API calls (GPT-4 narratives + questionnaires) ~5 minutes
Total:      ~8-10 minutes, ~40 API calls, ~$0.50
```

### Second Run (With Cache)
```
Embeddings:  0 API calls (all from cache) ~10 seconds
LLM:         0 API calls (all from cache) ~5 seconds
Total:       ~30 seconds, 0 API calls, $0.00
```

### Incremental Run (100 New Commits)
```
Embeddings:  1 API call (1 batch) ~10 seconds
LLM:         0 API calls (same clusters) ~5 seconds
Total:       ~1 minute, 1 API call, ~$0.01
```

## Error Handling

```
CacheManager
├─ File not found → logger.warning(), return empty dict
├─ JSON parse error → logger.warning(), skip row
└─ Write error → logger.error(), but continue execution

EmbeddingGenerator
├─ API error → logger.error(), propagate exception
└─ Batch too large → handled by batch_size = 100

LLMNarrativeGenerator
├─ API error → logger.error(), return placeholder
├─ JSON format error → logger.error(), return default
└─ Knowledge gaps missing → gracefully handle

All operations are wrapped in try-except with appropriate logging
```

## Security Considerations

- `.env` file is git-ignored (never committed)
- API key loaded from `.env` only at startup
- Cache files (CSV) contain only vectors and text, no credentials
- All file I/O uses standard Python libraries
- No sensitive data in logs (only metrics and progress)

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level
heManager
├─ File not found → logger.warning(), return empty dict
├─ JSON parse error → logger.warning(), skip row
└─ Write error → logger.error(), but continue execution

EmbeddingGenerator
├─ API error → logger.error(), propagate exception
└─ Batch too large → handled by batch_size = 100

LLMNarrativeGenerator
├─ API error → logger.error(), return placeholder
├─ JSON format error → logger.error(), return default
└─ Knowledge gaps missing → gracefully handle

All operations are wrapped in try-except with appropriate logging
```

## Security Considerations

- `.env` file is git-ignored (never committed)
- API key loaded from `.env` only at startup
- Cache files (CSV) contain only vectors and text, no credentials
- All file I/O uses standard Python libraries
- No sensitive data in logs (only metrics and progress)

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level

heManager
├─ File not found → logger.warning(), return empty dict
├─ JSON parse error → logger.warning(), skip row
└─ Write error → logger.error(), but continue execution

EmbeddingGenerator
├─ API error → logger.error(), propagate exception
└─ Batch too large → handled by batch_size = 100

LLMNarrativeGenerator
├─ API error → logger.error(), return placeholder
├─ JSON format error → logger.error(), return default
└─ Knowledge gaps missing → gracefully handle

All operations are wrapped in try-except with appropriate logging
```

## Security Considerations

- `.env` file is git-ignored (never committed)
- API key loaded from `.env` only at startup
- Cache files (CSV) contain only vectors and text, no credentials
- All file I/O uses standard Python libraries
- No sensitive data in logs (only metrics and progress)

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level

heManager
├─ File not found → logger.warning(), return empty dict
├─ JSON parse error → logger.warning(), skip row
└─ Write error → logger.error(), but continue execution

EmbeddingGenerator
├─ API error → logger.error(), propagate exception
└─ Batch too large → handled by batch_size = 100

LLMNarrativeGenerator
├─ API error → logger.error(), return placeholder
├─ JSON format error → logger.error(), return default
└─ Knowledge gaps missing → gracefully handle

All operations are wrapped in try-except with appropriate logging
```

## Security Considerations

- `.env` file is git-ignored (never committed)
- API key loaded from `.env` only at startup
- Cache files (CSV) contain only vectors and text, no credentials
- All file I/O uses standard Python libraries
- No sensitive data in logs (only metrics and progress)

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP levelheManager
├─ File not found → logger.warning(), return empty dict
├─ JSON parse error → logger.warning(), skip row
└─ Write error → logger.error(), but continue execution

EmbeddingGenerator
├─ API error → logger.error(), propagate exception
└─ Batch too large → handled by batch_size = 100

LLMNarrativeGenerator
├─ API error → logger.error(), return placeholder
├─ JSON format error → logger.error(), return default
└─ Knowledge gaps missing → gracefully handle

All operations are wrapped in try-except with appropriate logging
```

## Security Considerations

- `.env` file is git-ignored (never committed)
- API key loaded from `.env` only at startup
- Cache files (CSV) contain only vectors and text, no credentials
- All file I/O uses standard Python libraries
- No sensitive data in logs (only metrics and progress)

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level

## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level
## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level
## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level
## Scalability

Current system designed for:
- 2,000-5,000 commits per repository
- 50-150 clusters typically
- Top 15 narratives generated
- Single-threaded sequential processing

Future optimizations:
- Parallel embedding generation using batching
- Parallel LLM generation using threading/async
- SQLite cache backend instead of CSV
- Connection pooling for API calls
- Response caching at HTTP level
