def train_model(training_data):
    # Load or initialize your model here
    model = YourModelClass()

    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Number of epochs
    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in training_data:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(training_data.dataset)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

    return model