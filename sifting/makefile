%.o: %.cpp
	@$(CC) $(CPPFLAGS) -c $^ -o $@

CC = gcc

CPP_INCLUDE_FLAGS = -I./ -I./lib/ -isystem ./include -isystem ./include/glad/include

CPP_BASE_FLAGS = $(CPP_INCLUDE_FLAGS)            \
-std=c++20 -Ofast -pie -pthread              \
-Wall -Wextra -Weffc++                  \
-Waggressive-loop-optimizations -Wc++14-compat -Wmissing-declarations    \
-Wcast-align -Wchar-subscripts -Wconditionally-supported       \
-Wconversion -Wctor-dtor-privacy -Wempty-body -Wformat-nonliteral     \
-Wformat-security -Wformat-signedness -Wformat=2 -Wlogical-op      \
-Wnon-virtual-dtor -Wopenmp-simd -Woverloaded-virtual -Wpacked -Wpointer-arith  \
-Winit-self -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo    \
-Wstrict-null-sentinel -Wstrict-overflow=2 -Wsuggest-attribute=noreturn    \
-Wsuggest-override -Wswitch-default             \
-Wsync-nand -Wundef -Wunreachable-code -Wuseless-cast        \
-Wvariadic-macros -Wno-literal-suffix -Wno-missing-field-initializers    \
-Wno-narrowing -Wno-old-style-cast -Wno-varargs -Wstack-protector     \
-Wstack-usage=8192 -Wno-unused-parameter -Wno-deprecated-declarations    \
-Wno-missing-declarations

CPP_SANITIZER_FLAGS = -fcheck-new              \
-fsized-deallocation -fstack-protector -fstrict-overflow -flto-odr-type-merging  \
-fno-omit-frame-pointer -fPIE -fsanitize=address,bool,${strip       \
}bounds,enum,float-cast-overflow,float-divide-by-zero,${strip       \
}integer-divide-by-zero,leak,nonnull-attribute,null,object-size,return,${strip   \
}returns-nonnull-attribute,shift,signed-integer-overflow,undefined,${strip    \
}unreachable,vla-bound,vptr

CPP_DEBUG_FLAGS = -D _DEBUG -ggdb3

CPPFLAGS = $(CPP_BASE_FLAGS) $(CPP_DEBUG_FLAGS)

SIFTING = Sifting.o

ifeq ($(OS), Windows_NT)
	SIFTING_BUILD = ../build/Sifting.exe
else
	SIFTING_BUILD = ../build/Sifting
endif

Sifting: $(SIFTING)
	$(CC) $(SIFTING) -o $(SIFTING_BUILD)

py:
	python setup.py build_ext -i
