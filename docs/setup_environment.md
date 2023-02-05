# Как подключиться к удаленной машине
Машина крутиться на `172.28.163.21`, подключиться можно к ней через `make connect`.



# Как настроить окружение


## Troubleshooting

### Issue
Error: `Failed to unlock the collection!`

## Solution
Run in your terminal: `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
