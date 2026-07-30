# Useful commands

## Docker

Launching the VM on the server require this command : 

```bash
sudo docker compose -f docker-compose.vm.yml up -d --build
```

Stopping the VM : 
```bash
sudo docker compose -f docker-compose.vm.yml down
```

To consult the logs live of the web services: 
```bash
sudo docker compose -f docker-compose.vm.yml logs -f
```

To read logs of the aeyeweb app:
```bash
sudo docker exec aeyeweb tail -f ./logs/app.log
```

## Chacha access

To access chacha directly, connection should be made from the VM docker: 
```bash
# access to the VM
ssh jeremy.duc@153.109.8.48


cd apps/a-eye_web_dev_jeremy/

# access to chacha
sudo docker exec -it aeyeweb ssh chacha

# To observe running/pending jobs from Dance partition
watch -n 1 squeue -p Dance -u jaime.barrancohernandez
```