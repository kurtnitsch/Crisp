// crisp.proto
syntax = "proto3";

message CognitivePacket {
    string dest = 1;
    MessageType msg_type = 2;
    string sender = 3;
    string msg_id = 4;         // UUID
    string timestamp = 5;       // ISO 8601
    int32 hops = 6;
    map<string, bytes> payload = 7;  // msgpack serialized
    bytes binary_payload = 8;
    bytes vet_signature = 9;
    bytes vet_public_key = 10;
    int32 vet_epoch = 11;
}

enum MessageType {
    QUERY = 0;
    DATA = 1;
    EXEC = 2;
    COMMAND = 3;
    ACK = 4;
    ERROR = 5;
    UPDATE = 6;
    DISCOVER = 7;
    POND = 8;
    KIP_CLAIM = 9;
    KIP_VOTE = 10;
    KIP_QUERY = 11;
    KIP_RESPONSE = 12;
}

message KnowledgeClaim {
    string claim_id = 1;        // UUID
    string content = 2;
    string author_id = 3;
    string timestamp = 4;       // ISO 8601
    string expiration = 5;      // ISO 8601
    bytes vector = 6;           // BF16Vector data
    map<string, bool> votes = 7;
    KnowledgeClaimStatus status = 8;
    float confidence = 9;
    KnowledgeTrustLevel trust_level = 10;
    repeated string supporting_claims = 11;
    repeated string refuting_claims = 12;
}

enum KnowledgeClaimStatus {
    PENDING = 0;
    ACCEPTED = 1;
    REJECTED = 2;
    EXPIRED = 3;
}

enum KnowledgeTrustLevel {
    UNVERIFIED = 0;
    WEAK = 1;
    MODERATE = 2;
    STRONG = 3;
    VERIFIED = 4;
}
