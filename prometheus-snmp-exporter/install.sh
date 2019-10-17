helm upgrade --install prometheus-snmp-exporter -f values.yaml \
--set-file config=snmp.yml \
stable/prometheus-snmp-exporter
