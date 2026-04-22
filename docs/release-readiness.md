# Release Readiness Checklist

This checklist is **must-pass** before cutting any production release.

## Required gates

- [ ] **CI green**
  - All required workflows pass on the release commit (tests, lint, build).
  - No required check is in pending, neutral, or skipped state.

- [ ] **Security scan clean**
  - Dependency and static security scans report no unresolved high/critical findings.
  - Any accepted risk has a documented waiver with expiration date.

- [ ] **Migration validation**
  - Database/schema migrations run successfully in a production-like environment.
  - Forward migration and application startup verification are completed.

- [ ] **Rollback test**
  - A rollback procedure is executed in staging (application and schema where applicable).
  - Service health, data integrity, and core API smoke tests pass after rollback.

- [ ] **Load test baseline**
  - Load test results meet or improve baseline SLOs (latency, error rate, throughput).
  - Any regression has explicit sign-off and mitigation plan.

## Release sign-off

- [ ] Release manager sign-off
- [ ] Engineering sign-off
- [ ] Security sign-off (if changes impact threat model)

## Evidence links (attach per release)

- CI run URL:
- Security scan report URL:
- Migration validation log/artifact URL:
- Rollback drill notes URL:
- Load test report URL:
