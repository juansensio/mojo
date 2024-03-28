# Currently, Mojo doesn't support top-level code in a .mojo (or .🔥) file, so every program must include a function named main() as the entry point. You can declare it with either def or fn:

def greet(name):
    return "Hello, " + name + "!"

fn greet2(name: String) -> String:
    return "Hello, " + name + "!"

def main():
   print("Hello, world!")
   print(greet2("juan"))
   print(greet("ana"))

# fn main():
# 	print("Hello, world!")
# 	print(greet2("juan"))
# 	print(greet("ana"))


# el main puede ser declarado con def o fn
# puedo llamar a ambos tipos de funciones desde el cualquiera de los dos