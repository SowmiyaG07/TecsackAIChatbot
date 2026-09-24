/*
    TecsackAIChatbot - observability schema (SQL Server)

    Idempotent: safe to run more than once.

    Run:
        sqlcmd -S localhost -E -C -i sql\001_observability_schema.sql

    Design follows the existing MerchantAgentMonitoring
    ConversationHistory / LLMUsage tables, plus the RAG-specific
    retrieval and latency columns.
*/

-- Required for the filtered index (sqlcmd defaults QUOTED_IDENTIFIER OFF).
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF DB_ID(N'TecsackAIChatbot') IS NULL
    CREATE DATABASE TecsackAIChatbot;
GO

USE TecsackAIChatbot;
GO

/* ------------------------------------------------------------
   One row per chatbot request (one trace_id).
   ------------------------------------------------------------ */
IF OBJECT_ID(N'dbo.ConversationHistory', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ConversationHistory
    (
        id                          BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ConversationHistory PRIMARY KEY,

        trace_id                    UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT UQ_ConversationHistory_trace_id UNIQUE,

        -- NULL until the API layer supplies sessions.
        session_id                  NVARCHAR(100)    NULL,

        question                    NVARCHAR(MAX)    NOT NULL,
        answer                      NVARCHAR(MAX)    NULL,

        created_at                  DATETIME2(3)     NOT NULL,
        completed_at                DATETIME2(3)     NULL,

        model_name                  NVARCHAR(100)    NULL,

        -- Retrievers used and number of retriever invocations.
        tool_names                  NVARCHAR(200)    NULL,
        tool_call_count             INT              NOT NULL
            CONSTRAINT DF_ConversationHistory_tool_call_count DEFAULT (0),

        -- Retrieval
        query_intent                NVARCHAR(20)     NULL,
        expanded_query_count        INT              NULL,
        retrieved_chunk_count       INT              NULL,  -- unique candidates before top-K
        final_top_k_count           INT              NULL,  -- chunks sent to the LLM

        -- Latency (milliseconds)
        total_latency_ms            INT              NULL,
        retrieval_latency_ms        INT              NULL,  -- includes query expansion
        generation_latency_ms       INT              NULL,
        query_expansion_latency_ms  INT              NULL,
        embedding_latency_ms        INT              NULL,
        vector_search_latency_ms    INT              NULL,
        bm25_latency_ms             INT              NULL,

        -- Intent justification, expanded queries, top-K chunk ids, ...
        retrieval_details           NVARCHAR(MAX)    NULL
            CONSTRAINT CK_ConversationHistory_retrieval_details_json
            CHECK (retrieval_details IS NULL OR ISJSON(retrieval_details) = 1),

        result_type                 NVARCHAR(50)     NULL,  -- answered | not_found | no_results
        status                      NVARCHAR(20)     NOT NULL,  -- success | error
        error_message               NVARCHAR(MAX)    NULL
    );

    CREATE INDEX IX_ConversationHistory_created_at
        ON dbo.ConversationHistory (created_at);

    CREATE INDEX IX_ConversationHistory_session_id
        ON dbo.ConversationHistory (session_id)
        WHERE session_id IS NOT NULL;
END;
GO

/* ------------------------------------------------------------
   One row per LLM call (query_expansion, answer_generation, ...).
   ------------------------------------------------------------ */
IF OBJECT_ID(N'dbo.LLMUsage', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.LLMUsage
    (
        id              BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_LLMUsage PRIMARY KEY,

        trace_id        UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT FK_LLMUsage_ConversationHistory
            REFERENCES dbo.ConversationHistory (trace_id),

        call_purpose    NVARCHAR(50)     NOT NULL,
        model_name      NVARCHAR(100)    NOT NULL,

        -- NULL = not reported by the model server (never estimated).
        input_tokens    INT              NULL,
        output_tokens   INT              NULL,
        total_tokens    INT              NULL,

        latency_ms      INT              NULL,

        -- NULL unless LLM_*_COST_PER_1M_TOKENS are configured.
        estimated_cost  DECIMAL(18,8)    NULL,
        currency        CHAR(3)          NULL,

        status          NVARCHAR(20)     NOT NULL
            CONSTRAINT DF_LLMUsage_status DEFAULT (N'success'),

        created_at      DATETIME2(3)     NOT NULL
            CONSTRAINT DF_LLMUsage_created_at DEFAULT (SYSUTCDATETIME())
    );

    CREATE INDEX IX_LLMUsage_trace_id
        ON dbo.LLMUsage (trace_id);

    CREATE INDEX IX_LLMUsage_created_at
        ON dbo.LLMUsage (created_at);
END;
GO
