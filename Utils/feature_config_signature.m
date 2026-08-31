function signature = feature_config_signature(config)
%FEATURE_CONFIG_SIGNATURE Stable cache key for extraction-only settings.
extractConfig.ecg = config.ecg;
extractConfig.spo2 = config.spo2;
extractConfig.contextBeforeSeconds = config.contextBeforeSeconds;
extractConfig.contextAfterSeconds = config.contextAfterSeconds;
signature = jsonencode(extractConfig);
end
