-- MySQL 8.0+
-- Fingerprint-probe schema. Import feature definitions before importing the
-- fingerprint JSON documents that reference them.

CREATE TABLE IF NOT EXISTS fingerprint_prompts (
    prompt_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键，特征定义版本的唯一标识',
    prompt_id VARCHAR(128) NOT NULL COMMENT '探针的稳定业务标识，对应 PromptTemplate.id，例如 favorite_number',
    definition_hash CHAR(64) NOT NULL COMMENT '特征定义 SHA-256：prompt、变体、解析/检查函数版本、提取规则和验证策略的规范化哈希',
    category VARCHAR(128) NOT NULL COMMENT '探针业务分类，对应 PromptTemplate.category',
    prompt_variants JSON NOT NULL COMMENT '完整 prompt 变体数组，对应 PromptTemplate.variants；数组下标即 variant_index',
    system_prompt LONGTEXT NULL COMMENT '该探针附加的系统提示词，对应 PromptTemplate.system；实际请求还会合并项目默认系统提示词',
    feature_kind VARCHAR(64) NOT NULL COMMENT '特征类型，例如 parsed 或 length，对应 PromptTemplate.feature_kind',
    feature_field VARCHAR(255) NULL COMMENT '解析结果的点分字段路径，例如 parsed.score.value，对应 PromptTemplate.field',
    length_field VARCHAR(64) NULL COMMENT '长度特征的响应字段，例如 content 或 reasoning_content，对应 PromptTemplate.length_field',
    parser_reference VARCHAR(512) NOT NULL COMMENT '解析函数的可追溯引用，例如 vestigia.prompts.project_success_score.parse',
    checker_reference VARCHAR(512) NOT NULL COMMENT '检查函数的可追溯引用，例如 vestigia.prompts.project_success_score.check',
    parser_configuration JSON NOT NULL COMMENT '解析器所依赖的可配置常量和约束，例如数值范围、格式规则；无配置时保存空对象',
    checker_configuration JSON NOT NULL COMMENT '检查器所依赖的可配置常量和接受规则；无配置时保存空对象',
    stability_policy JSON NOT NULL COMMENT '稳定性验证策略：sample_size、resamples、seed、max_p95_tv_distance 和采用的距离度量',
    is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否为当前可用于新采样的特征定义；历史版本应置 0 而非删除',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '该特征定义写入数据库的时间',
    retired_at TIMESTAMP NULL DEFAULT NULL COMMENT '停止用于新采样的时间；NULL 表示尚未停用',

    PRIMARY KEY (prompt_id),
    UNIQUE KEY uq_fingerprint_prompts_definition (prompt_id, definition_hash),
    UNIQUE KEY uq_fingerprint_prompts_id_prompt (prompt_id, prompt_id),
    KEY ix_fingerprint_prompts_active (prompt_id, is_active)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci
  COMMENT='行为指纹特征（探针）定义及其版本；解析、检查和稳定性规则是定义的一部分';


CREATE TABLE IF NOT EXISTS model_fingerprints (
    fingerprint_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键，指纹记录唯一标识',

    -- Provenance and model/experiment identity.
    vendor VARCHAR(128) NOT NULL COMMENT '厂商或接入渠道名，取自 fingerprints 下的第一级目录',
    model VARCHAR(255) NOT NULL COMMENT '生成该指纹的模型名称，对应 JSON 的 model',
    prompt_id VARCHAR(128) NOT NULL COMMENT '冗余保存的探针标识，对应 JSON 的 prompt_id，便于按探针查询',
    parameters_hash CHAR(16) NOT NULL COMMENT '实验参数哈希，对应 JSON 的 parameters_hash',

    -- Probe definition selected during collection.
    prompt LONGTEXT NOT NULL COMMENT '本次采样实际使用的完整 prompt 文本，对应 JSON 的 prompt，应为特征定义某个变体',
    feature_kind VARCHAR(64) NOT NULL COMMENT '本次采样的特征提取类型，对应 JSON 的 feature_kind',
    feature_field VARCHAR(255) NOT NULL COMMENT '本次采样从响应中提取特征的字段路径，对应 JSON 的 field',

    -- Nested JSON fields from the fingerprint document. JSON preserves
    -- arbitrary response values and permits future schema additions.
    request_params JSON NOT NULL COMMENT '请求参数 JSON，例如 temperature、top_p、system_prompt',
    sample_values JSON NOT NULL COMMENT '原始采样特征值数组，对应 JSON 的 values',
    distribution JSON NOT NULL COMMENT '采样特征的经验概率分布，对应 JSON 的 distribution',
    stability JSON NOT NULL COMMENT '本次采样的重采样稳定性及距离统计结果，对应 JSON 的 stability',

    -- The fingerprint JSON timestamps are RFC 3339 local timestamps without
    -- a timezone suffix, so they are retained exactly rather than converted.
    started_at VARCHAR(40) NULL COMMENT '采样开始时间原文，对应 JSON 的 started_at',
    finished_at VARCHAR(40) NULL COMMENT '采样结束时间原文，对应 JSON 的 finished_at',
    imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '该记录写入数据库的时间',

    PRIMARY KEY (fingerprint_id),
    UNIQUE KEY uq_model_fingerprints_source_path (source_path),
    KEY ix_model_fingerprints_feature_lookup (parameters_hash, model),
    KEY ix_model_fingerprints_prompt_lookup (prompt_id, parameters_hash, model),
    KEY ix_model_fingerprints_vendor_model (vendor, model),
    KEY ix_model_fingerprints_started_at (started_at),
    CONSTRAINT fk_model_fingerprints_feature
        FOREIGN KEY (prompt_id) REFERENCES fingerprint_prompts (prompt_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci
  COMMENT='大语言模型行为指纹；一行对应 fingerprints 目录中的一个 JSON 指纹文件';

