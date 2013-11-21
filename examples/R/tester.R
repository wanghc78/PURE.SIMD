library(compiler)

run_add <- function(cons) {
	ret = 1

	a = 1:cons
	b = 1:cons
	ans = 1:cons
	c = 1:cons

	for (i in 1:cons) {
		a[i] = i
		b[i] = cons - i
		ans[i] = a[i] + b[i]
	}
	print(system.time(a + b))
	c = a + b

	for (i in 1:cons) {
		if (c[i] != ans[i]) {
			ret = 0
			break
		}
	}

	if (ret == 1) {
		print("CORRECT")
	}
	else {
		print("INCORRECT")
	}
}

run_sub <- function(cons) {
	ret = 1

	a = 1:cons
	b = 1:cons
	ans = 1:cons
	c = 1:cons

	for (i in 1:cons) {
		a[i] = i
		b[i] = cons - i
		ans[i] = a[i] - b[i]
	}
	print(system.time(a - b))
	c = a - b

	for (i in 1:cons) {
		if (c[i] != ans[i]) {
			ret = 0
			break
		}
	}

	if (ret == 1) {
		print("CORRECT")
	}
	else {
		print("INCORRECT")
	}
}

run_mul <- function(cons) {
	ret = 1

	a = 1:cons
	b = 1:cons
	ans = 1:cons
	c = 1:cons

	for (i in 1:cons) {
		a[i] = i
		b[i] = cons - i
		ans[i] = a[i] * b[i]
	}
	print(system.time(a * b))
	c = a * b

	for (i in 1:cons) {
		if (c[i] != ans[i]) {
			ret = 0
			break
		}
	}

	if (ret == 1) {
		print("CORRECT")
	}
	else {
		print("INCORRECT")
	}
}

run_div <- function(cons) {
	ret = 1

	a = 1:cons
	b = 1:cons
	ans = 1:cons
	c = 1:cons

	for (i in 1:cons) {
		a[i] = i
		b[i] = cons - i
		ans[i] = a[i] / b[i]
	}
	print(system.time(a / b))
	c = a / b

	for (i in 1:cons) {
		if (c[i] != ans[i]) {
			ret = 0
			break
		}
	}

	if (ret == 1) {
		print("CORRECT")
	}
	else {
		print("INCORRECT")
	}
}

run_addC <- cmpfun(run_add)
run_subC <- cmpfun(run_sub)
run_mulC <- cmpfun(run_mul)
run_divC <- cmpfun(run_div)
