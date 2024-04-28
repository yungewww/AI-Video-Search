CREATE TABLE IF NOT EXISTS image_vector (
    vidId INT,
    frameNum INT,
    timestamp FLOAT,
    detectedObjId INT,
    detectedObjClass TEXT,
    confidence FLOAT,
    bbox_info TEXT,
    vector TEXT
);

COPY image_vector(vidId, frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox_info, vector)
FROM '/app/image_vector.csv' CSV HEADER;
