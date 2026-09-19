import matplotlib.pyplot as plt
# Syntaxe de base absolue :
plt.plot(liste_des_epochs, liste_de_la_loss)
plt.title("Évolution de l'erreur")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show() # Ouvre la fenêtre graphique
