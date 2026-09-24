/*
    TecsackAIChatbot - SQL login for the chatbot application.

    Least privilege: the app can only INSERT observability rows.
    Idempotent: safe to run more than once (the password is only
    set when the login is created).

    Run as an administrator, passing the password as a variable
    (never commit the password):

        sqlcmd -S <server> -E -C -i sql\002_observability_app_login.sql -v APP_PASSWORD="<strong password>"

    Then set in .env:

        SQLSERVER_CONNECTION_STRING=Driver={ODBC Driver 18 for SQL Server};Server=<server>;Database=TecsackAIChatbot;UID=tecsack_chatbot_app;PWD=<strong password>;Encrypt=yes;TrustServerCertificate=no;
*/

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

USE master;
GO

IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = N'tecsack_chatbot_app')
    CREATE LOGIN tecsack_chatbot_app
        WITH PASSWORD = N'$(APP_PASSWORD)',
             DEFAULT_DATABASE = TecsackAIChatbot,
             CHECK_POLICY = ON,
             CHECK_EXPIRATION = OFF;
GO

USE TecsackAIChatbot;
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = N'tecsack_chatbot_app')
    CREATE USER tecsack_chatbot_app FOR LOGIN tecsack_chatbot_app;
GO

GRANT INSERT ON dbo.ConversationHistory TO tecsack_chatbot_app;
GRANT INSERT ON dbo.LLMUsage TO tecsack_chatbot_app;
GO
